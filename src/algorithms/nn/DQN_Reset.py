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
            state.steps % state.hypers.reset_steps == 0,
            self._reset,
            lambda s: s,
            state,
        )
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _reset(self, state: AgentState):
        key, q_key, body_key = jax.random.split(state.key, 3)
        optimizer = optax.adam(**state.hypers.optimizer.__dict__)
        params = state.params
        optim = state.optim

        q_params = self.q_net.init(q_key, self.builder._sample_phi)
        params["q"] = q_params
        optim["q"] = optimizer.init(q_params)

        def _reset_body():
            body_params = self.builder.reset(body_key)
            body_optim = optimizer.init(body_params["phi"])
            return body_params["phi"], body_optim

        def _no_reset_body():
            return params["phi"], optim["phi"]

        new_phi, new_optim_phi = jax.lax.cond(
            state.hypers.reset_head_only,
            _no_reset_body,
            _reset_body,
        )
        params["phi"] = new_phi
        optim["phi"] = new_optim_phi

        return replace(state, key=key, params=params, optim=optim)
