# ActorCriticCoordConv: PPO with CoordConv (appends x,y coordinate channels)
import flax.linen as nn
from typing import Optional, Tuple, Union, Any, Sequence, Dict
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import numpy as np
import distrax
import functools
from algorithms.nn.rtus.rtus import *


def _add_coord_channels(x: jnp.ndarray) -> jnp.ndarray:
    """Append normalised (x, y) coordinate channels to image tensor.

    Works for shapes (..., H, W, C).  Coordinates are in [-1, 1].
    """
    *batch, h, w, _c = x.shape
    y_coords = jnp.linspace(-1.0, 1.0, h)[:, None]          # (H, 1)
    x_coords = jnp.linspace(-1.0, 1.0, w)[None, :]          # (1, W)
    y_grid = jnp.broadcast_to(y_coords, (h, w))[..., None]   # (H, W, 1)
    x_grid = jnp.broadcast_to(x_coords, (h, w))[..., None]   # (H, W, 1)
    coords = jnp.concatenate([x_grid, y_grid], axis=-1)      # (H, W, 2)
    for _ in batch:
        coords = coords[None, ...]
    coords = jnp.broadcast_to(coords, (*batch, h, w, 2))
    return jnp.concatenate([x, coords], axis=-1)


class ActorCriticCoordConv(nn.Module):
    action_dim: int
    d_hidden: int = 192
    hidden_size: int = 64
    activation: str = "tanh"
    cont: bool = False
    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, hidden, obs):
        """
        hidden: Any
        obs: ((batch_size, H, W, C), (batch_size, action_dim), (batch_size, 1), ...)
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        (obs, last_action_encoded, last_reward, sine, cosine, reward_trace) = obs
        last_reward_plus = last_reward
        if self.use_sinusoidal_encoding:
            last_reward_plus = jnp.concatenate((last_reward_plus, sine, cosine), axis=-1)
        if self.use_reward_trace:
            last_reward_plus = jnp.concatenate((last_reward_plus, reward_trace), axis=-1)

        # Prepend coordinate channels: (B, H, W, C) → (B, H, W, C+2)
        obs_with_coords = _add_coord_channels(obs)

        # Actor path: CoordConv → Conv 3×3 on C+2 channels
        actor_embedding = nn.Conv(16, 3, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_conv1")(obs_with_coords)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm1")(actor_embedding)
        actor_embedding = activation(actor_embedding)
        actor_embedding = jnp.reshape(actor_embedding, (actor_embedding.shape[0], -1))
        actor_embedding = jnp.concatenate((actor_embedding, last_action_encoded, last_reward_plus), axis=-1)
        actor_embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_dense2")(actor_embedding)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm2")(actor_embedding)
        actor_embedding = activation(actor_embedding)

        # Critic path: CoordConv → Conv 3×3 on C+2 channels
        critic_embedding = nn.Conv(16, 3, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_conv1")(obs_with_coords)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm1")(critic_embedding)
        critic_embedding = activation(critic_embedding)
        critic_embedding = jnp.reshape(critic_embedding, (critic_embedding.shape[0], -1))
        critic_embedding = jnp.concatenate((critic_embedding, last_action_encoded, last_reward_plus), axis=-1)
        critic_embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_dense2")(critic_embedding)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm2")(critic_embedding)
        critic_embedding = activation(critic_embedding)

        actor_embedding = nn.Dense(self.d_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_dense3")(actor_embedding)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm3")(actor_embedding)
        actor_embedding = activation(actor_embedding)
        critic_embedding = nn.Dense(self.d_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_dense3")(critic_embedding)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm3")(critic_embedding)
        critic_embedding = activation(critic_embedding)

        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor_dense4")(actor_embedding)
        if self.use_layernorm:
            actor_mean = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm4")(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_mean")(actor_mean)
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic_dense4")(critic_embedding)
        if self.use_layernorm:
            critic = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm4")(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        return None
