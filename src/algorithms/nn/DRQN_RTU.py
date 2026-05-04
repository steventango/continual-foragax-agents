"""DRQN with a Linear RTU recurrent backbone trained via RTRL.

The recurrent core (`RTLRTUs` in src/algorithms/nn/rtus/rtus.py) ships with a
custom-VJP that replaces BPTT-through-history with the Real-Time Recurrent
Learning gradient. We reuse it directly inside a small Flax network, plug
that network into the existing `NNAgent`/`DRQN` machinery via the
`_setup_network` hook, and store the per-step (hidden, grad_memory) carry
pytree in the Flashbax replay buffer.
"""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Tuple, Union

import flax.linen as fnn
import jax
import jax.numpy as jnp
import optax
from flax.linen.initializers import constant, orthogonal
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DRQN import AgentState as DRQNAgentState
from algorithms.nn.DRQN import DRQN
from algorithms.nn.DRQN import Hypers as DRQNHypers
from algorithms.nn.DRQN import q_loss
from algorithms.nn.rtus.rtus import RTLRTUs


class DRQNRTUNet(fnn.Module):
    """Per-timestep DRQN backbone with a Linear RTU + RTRL.

    Forward signature: ``__call__(carry, obs, scalars) -> (new_carry, q)``
    where ``carry`` is the ``RTLRTUs`` pytree ``(hidden_init, memory_grad_init)``
    initialized via ``DRQNRTUNet.initial_carry``.
    """

    n_actions: int
    hidden_size: int = 64
    rtu_hidden: int = 32
    rtu_params_type: str = "exp_exp"
    rtu_activation: str = "relu"
    use_layernorm: bool = True
    pre_rtu_layers: int = 0
    post_rtu_layers: int = 0
    conv_channels: int = 16
    conv_kernel: int = 3

    def _activation(self, x):
        if self.rtu_activation == "relu":
            return fnn.relu
        elif self.rtu_activation == "tanh":
            return fnn.tanh
        else:
            raise NotImplementedError(self.rtu_activation)

    @fnn.compact
    def __call__(self, carry, obs, scalars):
        # obs: (B, H, W, C); scalars: (B, S)
        act = (
            fnn.relu if self.rtu_activation == "relu" else fnn.tanh
        )

        h = fnn.Conv(
            self.conv_channels,
            (self.conv_kernel, self.conv_kernel),
            (1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="conv",
        )(obs)
        if self.use_layernorm:
            h = fnn.LayerNorm(epsilon=1e-5, name="conv_ln")(h)
        h = act(h)
        h = jnp.reshape(h, (h.shape[0], -1))

        if scalars is not None and scalars.shape[-1] > 0:
            h = jnp.concatenate([h, scalars], axis=-1)

        h = fnn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="dense",
        )(h)
        if self.use_layernorm:
            h = fnn.LayerNorm(epsilon=1e-5, name="dense_ln")(h)
        h = act(h)

        for i in range(self.pre_rtu_layers):
            h = fnn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"pre_rtu_{i}",
            )(h)
            if self.use_layernorm:
                h = fnn.LayerNorm(epsilon=1e-5, name=f"pre_rtu_ln_{i}")(h)
            h = act(h)

        new_carry, rtu_out = RTLRTUs(
            n_hidden=self.rtu_hidden,
            params_type=self.rtu_params_type,
            d_input=self.hidden_size,
            activation=self.rtu_activation,
            name="rtu",
        )(carry, h)

        z = rtu_out
        for i in range(self.post_rtu_layers):
            z = fnn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"post_rtu_{i}",
            )(z)
            if self.use_layernorm:
                z = fnn.LayerNorm(epsilon=1e-5, name=f"post_rtu_ln_{i}")(z)
            z = act(z)

        q = fnn.Dense(
            self.n_actions,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="q_head",
        )(z)
        return new_carry, q

    @staticmethod
    def initial_carry(batch_size: int, rtu_hidden: int, d_input: int):
        h_init = (
            jnp.zeros((batch_size, rtu_hidden)),
            jnp.zeros((batch_size, rtu_hidden)),
        )
        mg_init = (
            jnp.zeros((batch_size, rtu_hidden)),
            jnp.zeros((batch_size, rtu_hidden)),
            jnp.zeros((batch_size, rtu_hidden)),
            jnp.zeros((batch_size, rtu_hidden)),
            jnp.zeros((batch_size, d_input, rtu_hidden)),
            jnp.zeros((batch_size, d_input, rtu_hidden)),
            jnp.zeros((batch_size, d_input, rtu_hidden)),
            jnp.zeros((batch_size, d_input, rtu_hidden)),
        )
        return (h_init, mg_init)


def _strip_leading_axis(carry):
    """Strip the leading length axis: (B, 1, ...) → (B, ...)."""
    return jax.tree.map(lambda c: jnp.squeeze(c, axis=1), carry)


def _expand_time_axis(carry):
    """Add a length-1 time axis: (B, ...) → (B, 1, ...)."""
    return jax.tree.map(lambda c: jnp.expand_dims(c, axis=1), carry)


def _slice_time(carry, start, stop):
    return jax.tree.map(lambda c: c[:, start:stop], carry)


def _split_time(carry, idx):
    head = jax.tree.map(lambda c: c[:, :idx], carry)
    tail = jax.tree.map(lambda c: c[:, idx:], carry)
    return head, tail


def _take_time(carry, t):
    return jax.tree.map(lambda c: c[:, t], carry)


def _set_time(carry, t, values):
    return jax.tree.map(lambda c, v: c.at[:, t].set(v), carry, values)


def _last_step(carry):
    return jax.tree.map(lambda c: c[:, -1], carry)


def _index0(carry):
    return jax.tree.map(lambda c: c[0], carry)


@cxu.dataclass
class Hypers(DRQNHypers):
    pass


@cxu.dataclass
class AgentState(DRQNAgentState):
    pass


class DRQN_RTU(DRQN):
    def _setup_network(self, image_shape):
        rep = self.rep_params
        self.rtu_hidden = int(rep.get("rtu_hidden", rep.get("hidden", 64) // 2))
        self.network = DRQNRTUNet(
            n_actions=self.actions,
            hidden_size=int(rep.get("hidden", 64)),
            rtu_hidden=self.rtu_hidden,
            rtu_params_type=rep.get("rtu_params_type", "exp_exp"),
            rtu_activation=rep.get("rtu_activation", "relu"),
            use_layernorm=bool(rep.get("use_layernorm", True)),
            pre_rtu_layers=int(rep.get("pre_rtu_layers", 0)),
            post_rtu_layers=int(rep.get("post_rtu_layers", 0)),
        )
        # Override hidden_size used by NNAgent for the buffer carry shape.
        # The carry is a pytree, but NNAgent.__init__ already finished using
        # self.hidden_size (only consumed by DRQN.__init__ for dummy_timestep,
        # which we override below).
        self.hidden_size = int(rep.get("hidden", 64))

        dummy_obs = jnp.zeros((1,) + tuple(image_shape))
        dummy_scalars = jnp.zeros((1, max(self.scalars_size, 1)))
        dummy_carry = DRQNRTUNet.initial_carry(1, self.rtu_hidden, self.hidden_size)

        self.key, init_key = jax.random.split(self.key)
        net_params = self.network.init(init_key, dummy_carry, dummy_obs, dummy_scalars)

        def phi(params, x, scalars=None, carry=None, reset=None, is_target=False):
            """Run the recurrent network across an (optional) time axis.

            Returns (q_seq, carry_seq, initial_carry) matching the layout DRQN
            expects from `self.phi` (positions 0, 1, 2 in `_values` and `_loss`).
            """
            # Add a length-1 time axis if absent (single-step calls during action).
            if x.ndim == 4:
                x = x[:, None]
            if scalars is None:
                scalars = jnp.zeros((x.shape[0], x.shape[1], 0))
            elif scalars.ndim == 2:
                scalars = scalars[:, None]

            B, T = x.shape[:2]
            zero_carry_b = DRQNRTUNet.initial_carry(B, self.rtu_hidden, self.hidden_size)

            if carry is None:
                init_carry = zero_carry_b
            else:
                # `carry` may be either an already-stripped (B, ...) carry or a
                # buffered sequence (B, T, ...). Take t=0 in the latter case.
                first_leaf = jax.tree.leaves(carry)[0]
                if first_leaf.ndim >= 3 and first_leaf.shape[1] == T:
                    init_carry = _take_time(carry, 0)
                else:
                    init_carry = carry

            if reset is None:
                reset = jnp.zeros((B, T), dtype=jnp.bool_)

            # Time-major scan inputs.
            x_tm = jnp.swapaxes(x, 0, 1)
            scalars_tm = jnp.swapaxes(scalars, 0, 1)
            reset_tm = jnp.swapaxes(reset, 0, 1)

            def body(running_carry, step_inputs):
                obs_i, scalars_i, reset_i = step_inputs
                # On reset, swap running carry for the zero pytree.
                running_carry = jax.tree.map(
                    lambda r, z: jnp.where(
                        reset_i.reshape((-1,) + (1,) * (r.ndim - 1)), z, r
                    ),
                    running_carry,
                    zero_carry_b,
                )
                next_carry, q_i = self.network.apply(
                    params, running_carry, obs_i, scalars_i
                )
                return next_carry, (next_carry, q_i)

            final_carry, (carry_tm, q_tm) = jax.lax.scan(
                body, init_carry, (x_tm, scalars_tm, reset_tm)
            )
            q_seq = jnp.swapaxes(q_tm, 0, 1)
            carry_seq = jax.tree.map(lambda c: jnp.swapaxes(c, 0, 1), carry_tm)

            initial_carry = DRQNRTUNet.initial_carry(
                1, self.rtu_hidden, self.hidden_size
            )
            return q_seq, carry_seq, initial_carry

        self.phi = phi
        # The Flax network already produces Q-values inside `phi`. Keep `q` as
        # an identity so DRQN's loss code (`self.q(params, phi)`) stays valid.
        self.q = lambda params, x: x
        self.initializers = {}
        return net_params

    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        # NNAgent and DRQN both populated `dummy_timestep["carry"]` with a
        # `jnp.zeros(self.hidden_size)` placeholder; replace it (and
        # re-init the buffer) with our RTU carry pytree so Flashbax learns
        # the right shapes.
        if isinstance(observations, Mapping):
            image_shape = observations["image"]
        else:
            image_shape = observations

        dummy_hint = None
        dummy_hint_trace = None
        if isinstance(observations, Mapping) and "hint" in observations and "hint" in self.scalar_features:
            dummy_hint = jnp.zeros(observations["hint"])
        if (
            isinstance(observations, Mapping)
            and "hint" in observations
            and "hint_trace" in self.scalar_features
        ):
            dummy_hint_trace = jnp.zeros(observations["hint"])

        zero_carry = _index0(
            DRQNRTUNet.initial_carry(1, self.rtu_hidden, self.hidden_size)
        )

        dummy_timestep = {
            "x": jnp.zeros(image_shape),
            "carry": zero_carry,
            "reset": jnp.bool_(True),
            "scalars": self.encode_scalar_features(
                jnp.int32(0),
                jnp.float32(0),
                jnp.float32(0),
                dummy_hint,
                dummy_hint_trace,
            ),
            "a": jnp.int32(0),
            "r": jnp.float32(0),
            "gamma": jnp.float32(0),
        }
        buffer_state = self.buffer.init(dummy_timestep)

        # Rewire AgentState to the freshly-initialised buffer + the
        # last_timestep dictionary (so subsequent `add` calls type-match).
        self.state = replace(
            self.state,
            buffer_state=buffer_state,
            last_timestep=dummy_timestep,
        )

    @partial(jax.jit, static_argnums=0)
    def _values(
        self,
        state: AgentState,
        x: jax.Array,
        scalars: jax.Array,
        carry=None,
    ):
        scalars_seq = jnp.expand_dims(scalars, 1)
        phi = self.phi(state.params, x, scalars=scalars_seq, carry=carry)
        return phi[0][:, -1], _last_step(phi[1]), phi[2]

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: Union[jax.Array, Dict[str, jax.Array]]):
        if isinstance(obs, Mapping):
            obs_img = obs["image"]
            hint = obs["hint"]
        else:
            obs_img = obs
            hint = None

        scalars = self.encode_scalar_features(
            jnp.int32(-1), jnp.float32(0), jnp.float32(0), hint
        )
        state, a = self.act(state, obs_img, scalars)

        zero_carry = _index0(
            DRQNRTUNet.initial_carry(1, self.rtu_hidden, self.hidden_size)
        )
        state.last_timestep.update(
            {
                "x": obs_img,
                "a": a,
                "scalars": scalars,
                "carry": zero_carry,
                "reset": jnp.bool_(True),
            }
        )
        state = self._decay_epsilon(state)
        state = self._maybe_update(state)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: Union[jax.Array, Dict[str, jax.Array]],
        extra: Dict[str, jax.Array],
    ):
        if isinstance(obs, Mapping):
            obs_img = obs["image"]
            hint = obs["hint"]
        else:
            obs_img = obs
            hint = None
        gamma = extra.get("gamma", 1.0)

        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(state.hypers.gamma * gamma),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        buffer_state = self.buffer.add(state.buffer_state, batch_sequence)
        state = replace(state, buffer_state=buffer_state)

        state, unbiased_reward_trace = self._compute_reward_trace(state, reward)

        scalars = self.encode_scalar_features(
            state.last_timestep["a"], reward, unbiased_reward_trace, hint
        )
        last_carry = _index0(state.carry)
        state, a = self.act(state, obs_img, scalars)

        state.last_timestep.update(
            {
                "x": obs_img,
                "a": a,
                "scalars": scalars,
                "carry": last_carry,
                "reset": jnp.bool_(False),
            }
        )
        state = self._maybe_update(state)
        state = self._decay_epsilon(state)
        return state, a

    def _loss(
        self, params, target, batch: Dict, weights: jax.Array
    ):
        B, T = batch["a"][:, :-1].shape
        weights = jnp.broadcast_to(weights[:, None], (B, T))

        x = batch["x"][:, :-1]
        xp = batch["x"][:, 1:]
        a = batch["a"][:, :-1]
        r = batch["r"][:, :-1]
        g = batch["gamma"][:, :-1]
        carry = _slice_time(batch["carry"], 0, -1)
        carryp = _slice_time(batch["carry"], 1, T + 1)
        reset = batch["reset"][:, :-1]

        scalars = batch["scalars"][:, :-1]
        scalars_p = batch["scalars"][:, 1:]

        if self.burn_in_steps > 0:
            b_x, x = jnp.split(x, [self.burn_in_steps], axis=1)
            b_xp, xp = jnp.split(xp, [self.burn_in_steps], axis=1)
            b_reset, reset = jnp.split(reset, [self.burn_in_steps], axis=1)
            b_carry, carry = _split_time(carry, self.burn_in_steps)
            b_carryp, carryp = _split_time(carryp, self.burn_in_steps)
            b_scalars, scalars = jnp.split(scalars, [self.burn_in_steps], axis=1)
            b_scalars_p, scalars_p = jnp.split(scalars_p, [self.burn_in_steps], axis=1)
            _, a = jnp.split(a, [self.burn_in_steps], axis=1)
            _, r = jnp.split(r, [self.burn_in_steps], axis=1)
            _, g = jnp.split(g, [self.burn_in_steps], axis=1)
            _, weights = jnp.split(weights, [self.burn_in_steps], axis=1)

            burn_carry_out = jax.lax.stop_gradient(
                _last_step(
                    self.phi(
                        params,
                        b_x,
                        scalars=b_scalars,
                        carry=b_carry,
                        reset=b_reset,
                        is_target=False,
                    )[1]
                )
            )
            burn_carryp_out = jax.lax.stop_gradient(
                _last_step(
                    self.phi(
                        target,
                        b_xp,
                        scalars=b_scalars_p,
                        carry=b_carryp,
                        reset=b_reset,
                        is_target=True,
                    )[1]
                )
            )
            carry = _set_time(carry, 0, burn_carry_out)
            carryp = _set_time(carryp, 0, burn_carryp_out)

        qs = self.phi(
            params, x, scalars=scalars, carry=carry, reset=reset, is_target=False
        )[0]
        qsp = self.phi(
            target, xp, scalars=scalars_p, carry=carryp, reset=reset, is_target=True
        )[0]

        qs = qs.reshape(-1, qs.shape[-1])
        qsp = qsp.reshape(-1, qsp.shape[-1])
        a = a.ravel()
        r = r.ravel()
        g = g.ravel()

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, batch_metrics = batch_loss(qs, a, r, g, qsp)

        loss = jnp.mean(losses)
        metrics = {
            "loss": loss,
            "abs_td_error": jnp.mean(jnp.abs(batch_metrics["delta"])),
            "squared_td_error": jnp.mean(jnp.square(batch_metrics["delta"])),
        }
        return loss, metrics
