# Modified from RealTimeACConv.py
# Variant 1: Equal-size vision / scalar embeddings.
#   vision   → Conv → flatten → Dense(hidden_size // 2)
#   scalars  → (action, reward, hint) concatenated → Dense(hidden_size // 2)
#   combined → concat → hidden_size → main RTU
# Used by PPO-RTU_LN_128_BALANCED and PPO-RTU_LN_128_HT

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.rtus.rtus import RTLRTUs, RTNLRTUs


class RealTimeActorCriticConvHint(nn.Module):
    """Like RealTimeActorCriticConv but projects the vision and scalar inputs
    (action, reward, hint) each into Dense(hidden_size // 2) so both
    branches produce equal-size embeddings before concatenation.
    The combined embedding (hidden_size) is then fed into the main RTU.
    The last_reward_plus input is [reward(1), hint(hint_dim)] concatenated.
    """

    action_dim: int
    d_hidden: int = 512
    hidden_size: int = 64
    activation: str = "tanh"
    cont: bool = False
    rtu_type: str = "linear_rtu"

    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    use_layernorm: bool = False
    conv: str = "PConv2DConv2D"

    @nn.compact
    def __call__(self, hidden, obs):
        """
        hidden: ((batch_size, d_hidden), (batch_size, d_hidden))
        obs: ((batch_size, H, W, C), (batch_size, action_dim),
              (batch_size, 1+hint_dim), sine, cosine, reward_trace)
              where last_reward = [reward(1), hint(hint_dim)]
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.rtu_type == "linear_rtu":
            seq_model = RTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = RTNLRTUs
        else:
            raise NotImplementedError

        (actor_hidden, critic_hidden) = hidden

        (obs, last_action_encoded, last_reward_plus, sine, cosine, reward_trace) = obs
        # Build scalar vector: [action, reward, hint, (optional sinusodal/rt)]
        scalars = jnp.concatenate((last_action_encoded, last_reward_plus), axis=-1)
        if self.use_sinusoidal_encoding:
            scalars = jnp.concatenate((scalars, sine, cosine), axis=-1)
        if self.use_reward_trace:
            scalars = jnp.concatenate((scalars, reward_trace), axis=-1)

        half = self.hidden_size // 2

        # ---- Actor vision branch ----
        actor_vis = obs
        if self.conv in ("PConv2D", "PConv2DConv2D"):
            actor_vis = nn.Conv(
                16,
                1,
                1,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="actor_pconv1",
            )(actor_vis)
            if self.use_layernorm:
                actor_vis = nn.LayerNorm(epsilon=1e-05, name="actor_player_norm1")(
                    actor_vis
                )
            actor_vis = activation(actor_vis)
        if self.conv in ("Conv2D", "PConv2DConv2D"):
            actor_vis = nn.Conv(
                16,
                3,
                1,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="actor_conv1",
            )(actor_vis)
            if self.use_layernorm:
                actor_vis = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm1")(
                    actor_vis
                )
            actor_vis = activation(actor_vis)
        actor_vis = jnp.reshape(actor_vis, (actor_vis.shape[0], -1))
        actor_vis = nn.Dense(
            half,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense2",
        )(actor_vis)
        if self.use_layernorm:
            actor_vis = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm2")(actor_vis)
        actor_vis = activation(actor_vis)  # (batch, half)

        # ---- Actor scalar branch (action + reward + hint) ----
        actor_scalars = nn.Dense(
            half,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_scalar_dense",
        )(scalars)
        if self.use_layernorm:
            actor_scalars = nn.LayerNorm(epsilon=1e-05, name="actor_scalar_layernorm")(
                actor_scalars
            )
        actor_scalars = activation(actor_scalars)  # (batch, half)

        # ---- Combine vision + scalars ----
        actor_embedding = jnp.concatenate(
            (actor_vis, actor_scalars), axis=-1
        )  # (batch, hidden_size)
        actor_embedding_skip = actor_embedding

        # ---- Critic vision branch ----
        critic_vis = obs
        if self.conv in ("PConv2D", "PConv2DConv2D"):
            critic_vis = nn.Conv(
                16,
                1,
                1,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="critic_pconv1",
            )(critic_vis)
            if self.use_layernorm:
                critic_vis = nn.LayerNorm(epsilon=1e-05, name="critic_player_norm1")(
                    critic_vis
                )
            critic_vis = activation(critic_vis)
        if self.conv in ("Conv2D", "PConv2DConv2D"):
            critic_vis = nn.Conv(
                16,
                3,
                1,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="critic_conv1",
            )(critic_vis)
            if self.use_layernorm:
                critic_vis = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm1")(
                    critic_vis
                )
            critic_vis = activation(critic_vis)
        critic_vis = jnp.reshape(critic_vis, (critic_vis.shape[0], -1))
        critic_vis = nn.Dense(
            half,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense2",
        )(critic_vis)
        if self.use_layernorm:
            critic_vis = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm2")(
                critic_vis
            )
        critic_vis = activation(critic_vis)  # (batch, half)

        # ---- Critic scalar branch (action + reward + hint) ----
        critic_scalars = nn.Dense(
            half,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_scalar_dense",
        )(scalars)
        if self.use_layernorm:
            critic_scalars = nn.LayerNorm(
                epsilon=1e-05, name="critic_scalar_layernorm"
            )(critic_scalars)
        critic_scalars = activation(critic_scalars)  # (batch, half)

        # ---- Combine vision + scalars ----
        critic_embedding = jnp.concatenate(
            (critic_vis, critic_scalars), axis=-1
        )  # (batch, hidden_size)
        critic_embedding_skip = critic_embedding

        actor_hidden, actor_embedding = seq_model(
            self.d_hidden, params_type="exp_exp", name="actor_rtu"
        )(actor_hidden, actor_embedding)
        critic_hidden, critic_embedding = seq_model(
            self.d_hidden, params_type="exp_exp", name="critic_rtu"
        )(critic_hidden, critic_embedding)

        actor_embedding = jnp.concatenate(
            (actor_embedding, actor_embedding_skip), axis=-1
        )
        critic_embedding = jnp.concatenate(
            (critic_embedding, critic_embedding_skip), axis=-1
        )

        actor_mean = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_dense3",
        )(actor_embedding)
        if self.use_layernorm:
            actor_mean = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm3")(
                actor_mean
            )
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_mean",
        )(actor_mean)
        if self.cont:
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_dense3",
        )(critic_embedding)
        if self.use_layernorm:
            critic = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm3")(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_value",
        )(critic)

        hidden = (actor_hidden, critic_hidden)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        """d_input should equal hidden_size (concat of two hidden_size//2 branches)."""
        actor_hidden_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
        )
        actor_memory_grad_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
        )
        critic_hidden_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
        )
        critic_memory_grad_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
        )
        carry_init = (
            (actor_hidden_init, actor_memory_grad_init),
            (critic_hidden_init, critic_memory_grad_init),
        )
        return carry_init
