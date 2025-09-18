import operator
from dataclasses import replace
from functools import partial
from typing import Any, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers


@cxu.dataclass
class Hypers(BaseHypers):
    lambda_l2_init: float


@cxu.dataclass
class AgentState(BaseAgentState):
    initial_params: Any
    hypers: Hypers


class DQN_L2_Init(DQN):
    def __init__(
        self,
        observations: tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        hypers = Hypers(
            **self.state.hypers.__dict__,
            lambda_l2_init=params["lambda_l2_init"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            initial_params=self.state.params,
            hypers=hypers,
        )

    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict):
        grad_fn = jax.grad(self._loss_w0, has_aux=True)
        grad, metrics = grad_fn(
            state.params,
            state.target_params,
            state.initial_params,
            batch,
            state.hypers.lambda_l2_init,
        )
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

    def _loss_w0(
        self,
        params: hk.Params,
        target: hk.Params,
        initial_params: hk.Params,
        batch: Dict,
        lambda_l2_init: float,
    ):
        q_l, metrics = super()._loss(params, target, batch)
        w0_loss = optax.l2_loss(
            ravel_pytree(params)[0], ravel_pytree(initial_params)[0]
        ).sum()
        reg_loss = lambda_l2_init * w0_loss
        total_loss = q_l + reg_loss
        return total_loss, metrics
