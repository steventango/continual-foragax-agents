from dataclasses import replace
from functools import partial
from typing import Dict, Tuple

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
    lambda_l2: float


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


class DQN_L2(DQN):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        hypers = Hypers(
            **self.state.hypers.__dict__,
            lambda_l2=params["lambda_l2"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict):
        grad_fn = jax.grad(self._loss_l2, has_aux=True)
        grad, metrics = grad_fn(
            state.params,
            state.target_params,
            batch,
            state.hypers.lambda_l2,
        )
        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)

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

    def _loss_l2(
        self,
        params: hk.Params,
        target: hk.Params,
        batch: Dict,
        lambda_l2: float,
    ):
        q_loss, metrics = super()._loss(params, target, batch)
        # Add L2 regularization penalty
        flat_params, _ = ravel_pytree(params)
        l2_penalty = optax.l2_loss(flat_params).sum()
        reg_loss = lambda_l2 * l2_penalty
        total_loss = q_loss + reg_loss
        return total_loss, metrics
