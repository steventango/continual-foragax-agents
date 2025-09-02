from functools import partial
from typing import Any, Dict, Tuple
from dataclasses import replace

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.NNAgent import AgentState as BaseAgentState
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import huber_loss


@cxu.dataclass
class AgentState(BaseAgentState):
    target_params: Any


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        "delta": delta,
    }


class DQN(NNAgent):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        # set up the target network parameters
        self.target_refresh = params["target_refresh"]

        self.state = AgentState(
            **self.state.__dict__, target_params=self.state.params
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        self.q = builder.addHead(lambda: hk.Linear(self.actions, name="q"))

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):  # type: ignore
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    def update(self):
        self.state = self._maybe_update(self.state)

    @partial(jax.jit, static_argnums=0)
    def _maybe_update(self, state: AgentState):
        # only update every `update_freq` steps
        # skip updates if the buffer isn't full yet
        return jax.lax.cond(
            (state.steps % self.update_freq == 0)
            & state.buffer.can_sample(state.buffer_state),
            lambda: self._update(state),
            lambda: state,
        )

    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        updates = state.updates + 1

        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = state.buffer.sample(state.buffer_state, buffer_sample_key)

        state, metrics = self._computeUpdate(
            state, batch.experience, batch.probabilities
        )

        priorities = metrics["delta"]
        buffer_state = state.buffer.set_priorities(
            state.buffer_state, batch.indices, priorities
        )

        target_params = jax.lax.cond(
            updates % self.target_refresh == 0,
            lambda: state.params,
            lambda: state.target_params,
        )

        return replace(
            state,
            updates=updates,
            buffer_state=buffer_state,
            target_params=target_params,
        )

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch, weights)

        updates, optim = state.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        return replace(state, params=params, optim=optim), metrics

    def _loss(
        self, params: hk.Params, target: hk.Params, batch: Dict, weights: jax.Array
    ):
        x = batch["x"][:, 0]
        xp = batch["x"][:, -1]
        a = batch["a"][:, 0]
        rs = batch["r"]
        gs = batch["gamma"]
        gs = jnp.concatenate([jnp.ones((gs.shape[0], 1)), gs[:, :-1]], axis=1)
        gs = jnp.cumprod(gs, axis=1)

        r = jnp.sum(rs[:, :-1] * gs[:, :-1], axis=1)
        g = gs[:, -1]

        phi = self.phi(params, x).out
        phi_p = self.phi(target, xp).out

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, metrics = batch_loss(qs, a, r, g, qsp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)

        return loss, metrics
