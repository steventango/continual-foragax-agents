from dataclasses import replace
from functools import partial
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
import utils.hk as hku
from algorithms.nn.NNAgent import (
    AgentState as NNAgentState,
)
from algorithms.nn.NNAgent import (
    Hypers as NNAgentHypers,
)
from algorithms.nn.NNAgent import (
    NNAgent,
)
from representations.networks import NetworkBuilder
from utils.jax import argmax_with_random_tie_breaking, vmap_except

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map


@cxu.dataclass
class Hypers(NNAgentHypers):
    beta: float


@cxu.dataclass
class AgentState(NNAgentState):
    hypers: Hypers


class EQRC(NNAgent):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        beta = params.get("beta", 1.0)

        hypers = Hypers(**self.state.hypers.__dict__, beta=beta)
        self.state = replace(self.state, hypers=hypers)

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        zero_init = hk.initializers.Constant(0)
        self.q = builder.addHead(
            lambda: hku.DuelingHeads(
                self.actions, name="q", w_init=zero_init, b_init=zero_init
            )
        )
        self.h = builder.addHead(
            lambda: hku.DuelingHeads(
                self.actions, name="h", w_init=zero_init, b_init=zero_init
            ),
            grad=False,
        )

    # jit'ed internal value function approximator
    # considerable speedup, especially for larger networks (note: haiku networks are not jit'ed by default)
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):  # type: ignore
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, buffer_sample_key)
        state, metrics = self._computeUpdate(state, batch.experience)

        priorities = metrics["delta"]
        buffer_state = self.buffer.set_priorities(
            state.buffer_state, batch.indices, priorities
        )

        return replace(state, buffer_state=buffer_state, updates=state.updates + 1)

    # -------------
    # -- Updates --
    # -------------

    # compute the update and return the new parameter states
    # and optimizer state (i.e. ADAM moving averages)
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict):
        params = state.params
        hypers = state.hypers
        grad, metrics = jax.grad(self._loss, has_aux=True)(
            params, hypers.epsilon, batch
        )
        optimizer = optax.adam(**hypers.optimizer.__dict__)
        updates, new_optim = optimizer.update(grad, state.optim, params)
        assert isinstance(updates, dict)

        decay = tree_map(
            lambda h, dh: dh - hypers.optimizer.learning_rate * hypers.beta * h,
            params["h"],
            updates["h"],  # type: ignore
        )

        updates |= {"h": decay}
        new_params = optax.apply_updates(params, updates)

        return replace(state, params=new_params, optim=new_optim), metrics

    # compute the total QRC loss for both sets of parameters (value parameters and h parameters)
    def _loss(self, params, epsilon, batch: Dict):
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
        q = self.q(params, phi)
        h = self.h(params, phi)

        phi_p = self.phi(params, xp).out
        qp = self.q(params, phi_p)

        # apply qc loss function to each sample in the minibatch
        # gives back value of the loss individually for parameters of v and h
        # note QC instead of QRC (i.e. no regularization)
        v_loss, h_loss, metrics = qc_loss(q, a, r, g, qp, h, epsilon)

        h_loss = h_loss.mean()
        v_loss = v_loss.mean()

        metrics |= {
            "v_loss": v_loss,
            "h_loss": h_loss,
        }

        return v_loss + h_loss, metrics


# ---------------
# -- Utilities --
# ---------------


@partial(vmap_except, exclude=["epsilon"])
def qc_loss(q, a, r, gamma, qtp1, h, epsilon):
    pi = argmax_with_random_tie_breaking(qtp1)

    pi = (1.0 - epsilon) * pi + (epsilon / qtp1.shape[0])
    pi = jax.lax.stop_gradient(pi)

    vtp1 = qtp1.dot(pi)
    target = r + gamma * vtp1
    target = jax.lax.stop_gradient(target)

    delta = target - q[a]
    delta_hat = h[a]

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * vtp1
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat) ** 2

    return (
        v_loss,
        h_loss,
        {
            "delta": delta,
            "h": delta_hat,
        },
    )
