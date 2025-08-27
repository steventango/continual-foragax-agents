import operator
from functools import partial
from typing import Any, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState


@cxu.dataclass
class AgentState(BaseAgentState):
    initial_params: Any


class W0_DQN(DQN):
    def __init__(
        self,
        observations: tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.w0_regularization = params['w0_regularization']

        self.state = AgentState(
            initial_params=self.state.params,
            params=self.state.params,
            target_params=self.state.target_params,
            buffer_state=self.state.buffer_state,
            optim=self.state.optim,
            key=self.state.key,
            last_timestep=self.state.last_timestep,
            steps=self.state.steps,
            updates=self.state.updates,
        )

    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict, weights: jax.Array):
        grad_fn = jax.grad(self._loss_w0, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, state.initial_params, batch, weights)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            target_params=state.target_params,
            buffer_state=state.buffer_state,
            optim=optim,
            key=state.key,
            last_timestep=state.last_timestep,
            steps=state.steps,
            updates=state.updates,
            initial_params=state.initial_params,
        )

        return new_state, metrics

    def _loss_w0(
        self, params: hk.Params, target: hk.Params, initial_params: hk.Params, batch: Dict, weights: jax.Array
    ):
        q_l, metrics = super()._loss(params, target, batch, weights)

        w0_loss = jax.tree.reduce(
            operator.add,
            jax.tree.map(
                lambda p, ip: jnp.sum(jnp.square(p - ip)),
                params,
                initial_params,
            ),
        )
        reg_loss = self.w0_regularization * w0_loss
        total_loss = q_l + reg_loss

        return total_loss, metrics
