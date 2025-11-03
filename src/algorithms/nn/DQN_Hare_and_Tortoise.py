from dataclasses import replace
from functools import partial
from typing import Dict

import jax
import jax.lax
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers


@cxu.dataclass
class Hypers(BaseHypers):
    ht_steps: int
    tau: float


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


class DQN_Hare_and_Tortoise(DQN):
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
            ht_steps=params["ht_steps"],
            tau=params["tau"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

    @partial(jax.jit, static_argnums=0)
    def _update_target_network(self, state: AgentState, updates: int):
        tau = state.hypers.tau
        target_params = jax.tree_util.tree_map(
            lambda t, o: (1 - tau) * t + tau * o, state.target_params, state.params
        )
        return target_params

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: jax.Array,
        extra: Dict[str, jax.Array],
    ):
        state, a = super()._step(state, reward, obs, extra)
        state = jax.lax.cond(
            state.steps % state.hypers.ht_steps == 0,
            self._reset,
            lambda s: s,
            state,
        )
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _reset(self, state: AgentState):
        params = state.target_params
        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)
        optim = optimizer.init(params)
        return replace(state, params=params, optim=optim)
