from dataclasses import replace
from functools import partial
from typing import Any, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.NNAgent import AgentState as BaseAgentState
from algorithms.nn.NNAgent import Hypers as BaseHypers
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import huber_loss


@cxu.dataclass
class Hypers(BaseHypers):
    target_refresh: int
    replacement_rate: float
    maturity_threshold: int


@cxu.dataclass
class AgentState(BaseAgentState):
    target_params: Any
    hypers: Hypers


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        "delta": delta,
    }


class DQN_CBP(NNAgent):
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
        hypers = Hypers(
            **self.state.hypers.__dict__,
            target_refresh=params["target_refresh"],
            replacement_rate=params["replacement_rate"],
            maturity_threshold=params["maturity_threshold"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            target_params=self.state.params,
            hypers=hypers,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        def q_net_builder():
            return hk.Linear(self.actions, name="q")

        self.q_net, _, self.q = builder.addHead(q_net_builder, name="q")

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):  # type: ignore
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        updates = state.updates + 1

        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, buffer_sample_key)

        state, metrics = self._computeUpdate(state, batch.experience)

        target_params = self._update_target_network(state, updates)

        return replace(
            state,
            updates=updates,
            target_params=target_params,
        ), metrics

    @partial(jax.jit, static_argnums=0)
    def _update_target_network(self, state: AgentState, updates: int):
        target_params = jax.lax.cond(
            updates % state.hypers.target_refresh == 0,
            lambda: state.params,
            lambda: state.target_params,
        )
        return target_params

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch)
        optimizer = optax.adam(**state.hypers.optimizer.__dict__)

        new_params = {}
        new_optim = {}
        weight_change = 0
        for name, p in state.params.items():
            updates, optim = optimizer.update(grad[name], state.optim[name], p)
            new_params[name] = optax.apply_updates(p, updates)
            new_optim[name] = optim
            flat_updates, _ = ravel_pytree(updates)
            weight_change += jnp.linalg.norm(flat_updates, ord=1)
        metrics["weight_change"] = weight_change

        return replace(state, params=new_params, optim=new_optim), metrics

    def _loss(self, params: hk.Params, target: hk.Params, batch: Dict):
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
        losses, batch_metrics = batch_loss(qs, a, r, g, qsp)

        loss = jnp.mean(losses)

        # aggregate metrics
        metrics = {
            "loss": loss,
            "abs_td_error": jnp.mean(jnp.abs(batch_metrics["delta"])),
            "squared_td_error": jnp.mean(jnp.square(batch_metrics["delta"])),
        }

        return loss, metrics
