from dataclasses import replace
from functools import partial
from typing import Dict

import jax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers
from representations.networks import NetworkBuilder


@cxu.dataclass
class Hypers(BaseHypers):
    reset_steps: int
    reset_head_only: bool


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


class DQN_Reset(DQN):
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
            reset_steps=params["reset_steps"],
            reset_head_only=params["reset_head_only"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

        self.builder = NetworkBuilder(observations, self.rep_params, seed)
        self._build_heads(self.builder)

    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        state = super()._update(state)

        state = jax.lax.cond(
            state.steps % state.hypers.reset_steps == 0,
            self._reset,
            lambda s: s,
            state,
        )

        return state

    def _reset(self, state: AgentState):
        key, subkey = jax.random.split(state.key)
        builder = NetworkBuilder(self.observations, self.rep_params, subkey)
        self._build_heads(builder)
        reset_params = builder.getParams()

        if state.hypers.reset_head_only:
            params = state.params.copy()
            params["q"] = reset_params["q"]
            for key in params["phi"].keys():
                if key == "phi_1" or key == "phi":
                    continue
                params["phi"][key] = reset_params["phi"][key]
        else:
            params = reset_params

        return replace(state, key=key, params=params)
