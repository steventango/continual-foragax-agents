# Modified from RealTimeACConv.py
# Variant 2: Only the hint has a dedicated RTU (d_hidden).
#   vision   → Conv → flatten → Dense(hidden_size) → vis_emb
#   hint     → RTU(d_hidden)                → hint_emb
#   combined → concat(vis_emb, hint_emb) → Dense(hidden_size) → output heads
# There is NO main RTU on the combined stream.
# Used by PPO-RTU_LN_128_HINT-RTU
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.rtus.rtus import RTLRTUs, RTNLRTUs


class RealTimeActorCriticConvHintRTU(nn.Module):
    """Like RealTimeActorCriticConv but only the hint passes through an RTU
    (d_hidden).  The vision embedding and hint RTU output are concatenated
    and fed directly to the output heads (no main RTU).

    Hidden state structure (per call):
        hidden = (actor_hint_carry, critic_hint_carry)
    where each *_carry is the RTU carry tuple (hidden_init, memory_grad_init).
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
        hidden: (actor_hint_carry, critic_hint_carry)
        obs: ((batch_size, H, W, C), (batch_size, action_dim),
              (batch_size, 1+hint_dim), sine, cosine, reward_trace)
              where last_reward_plus = [reward(1), hint(hint_dim)]
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

        (actor_hint_carry, critic_hint_carry) = hidden

        (obs, last_action_encoded, last_reward_plus, sine, cosine, reward_trace) = obs
        # last_reward_plus = [reward(1), hint(hint_dim)]
        reward = last_reward_plus[..., :1]   # (batch, 1)
        hint = last_reward_plus[..., 1:]     # (batch, hint_dim)

        # Build scalar vector for vision Dense: action + reward (+ optional)
        scalars = jnp.concatenate((last_action_encoded, reward), axis=-1)
        if self.use_sinusoidal_encoding:
            scalars = jnp.concatenate((scalars, sine, cosine), axis=-1)
        if self.use_reward_trace:
            scalars = jnp.concatenate((scalars, reward_trace), axis=-1)

        # ---- Actor vision branch ----
        actor_vis = obs
        if self.conv in ("PConv2D", "PConv2DConv2D"):
            actor_vis = nn.Conv(
                16, 1, 1,
                kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name="actor_pconv1",
            )(actor_vis)
            if self.use_layernorm:
                actor_vis = nn.LayerNorm(epsilon=1e-05, name="actor_player_norm1")(actor_vis)
            actor_vis = activation(actor_vis)
        if self.conv in ("Conv2D", "PConv2DConv2D"):
            actor_vis = nn.Conv(
                16, 3, 1,
                kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name="actor_conv1",
            )(actor_vis)
            if self.use_layernorm:
                actor_vis = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm1")(actor_vis)
            actor_vis = activation(actor_vis)
        actor_vis = jnp.reshape(actor_vis, (actor_vis.shape[0], -1))
        actor_vis = jnp.concatenate((actor_vis, scalars), axis=-1)
        actor_vis = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="actor_dense2",
        )(actor_vis)
        if self.use_layernorm:
            actor_vis = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm2")(actor_vis)
        actor_vis = activation(actor_vis)  # (batch, hidden_size)

        # ---- Actor hint RTU ----
        actor_hint_carry, actor_hint_emb = seq_model(
            self.d_hidden, params_type="exp_exp", name="actor_hint_rtu"
        )(actor_hint_carry, hint)

        # ---- Combine vision + hint RTU output (no main RTU) ----
        actor_embedding = jnp.concatenate(
            (actor_vis, actor_hint_emb), axis=-1
        )  # (batch, hidden_size + d_hidden)

        # ---- Critic vision branch ----
        critic_vis = obs
        if self.conv in ("PConv2D", "PConv2DConv2D"):
            critic_vis = nn.Conv(
                16, 1, 1,
                kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name="critic_pconv1",
            )(critic_vis)
            if self.use_layernorm:
                critic_vis = nn.LayerNorm(epsilon=1e-05, name="critic_player_norm1")(critic_vis)
            critic_vis = activation(critic_vis)
        if self.conv in ("Conv2D", "PConv2DConv2D"):
            critic_vis = nn.Conv(
                16, 3, 1,
                kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name="critic_conv1",
            )(critic_vis)
            if self.use_layernorm:
                critic_vis = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm1")(critic_vis)
            critic_vis = activation(critic_vis)
        critic_vis = jnp.reshape(critic_vis, (critic_vis.shape[0], -1))
        critic_vis = jnp.concatenate((critic_vis, scalars), axis=-1)
        critic_vis = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="critic_dense2",
        )(critic_vis)
        if self.use_layernorm:
            critic_vis = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm2")(critic_vis)
        critic_vis = activation(critic_vis)  # (batch, hidden_size)

        # ---- Critic hint RTU ----
        critic_hint_carry, critic_hint_emb = seq_model(
            self.d_hidden, params_type="exp_exp", name="critic_hint_rtu"
        )(critic_hint_carry, hint)

        # ---- Combine vision + hint RTU output (no main RTU) ----
        critic_embedding = jnp.concatenate(
            (critic_vis, critic_hint_emb), axis=-1
        )  # (batch, hidden_size + d_hidden)

        # ---- Actor output head ----
        actor_mean = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2), bias_init=constant(0.0),
            name="actor_dense3",
        )(actor_embedding)
        if self.use_layernorm:
            actor_mean = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm3")(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor_mean",
        )(actor_mean)
        if self.cont:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        # ---- Critic output head ----
        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2), bias_init=constant(0.0),
            name="critic_dense3",
        )(critic_embedding)
        if self.use_layernorm:
            critic = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm3")(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="critic_value",
        )(critic)

        hidden = (actor_hint_carry, critic_hint_carry)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        """
        d_hidden  : hint RTU hidden size
        hint_dim  : hint observation dimension (RTU input size)
        d_input   : unused (kept for API compatibility)
        """
        def _rtu_carry(d_h, d_i):
            h_init = (
                jnp.zeros((batch_size, d_h)),
                jnp.zeros((batch_size, d_h)),
            )
            mg_init = (
                jnp.zeros((batch_size, d_h)),
                jnp.zeros((batch_size, d_h)),
                jnp.zeros((batch_size, d_h)),
                jnp.zeros((batch_size, d_h)),
                jnp.zeros((batch_size, d_i, d_h)),
                jnp.zeros((batch_size, d_i, d_h)),
                jnp.zeros((batch_size, d_i, d_h)),
                jnp.zeros((batch_size, d_i, d_h)),
            )
            return (h_init, mg_init)

        actor_hint_carry  = _rtu_carry(d_hidden, d_input)
        critic_hint_carry = _rtu_carry(d_hidden, d_input)

        return (actor_hint_carry, critic_hint_carry)
