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
    sp_steps: int
    shrink_factor: float
    noise_scale: float


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


class DQN_Shrink_and_Perturb(DQN):
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
            sp_steps=params["sp_steps"],
            shrink_factor=params["shrink_factor"],
            noise_scale=params["noise_scale"],
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
            (state.steps % state.hypers.sp_steps == 0) & (state.steps > 0),
            self._shrink_and_perturb,
            lambda s: s,
            state,
        )
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _shrink_and_perturb(self, state: AgentState):
        key, subkey = jax.random.split(state.key)

        def sp(p):
            noise = jax.random.normal(subkey, shape=p.shape, dtype=p.dtype)
            return p * state.hypers.shrink_factor + noise * state.hypers.noise_scale

        params = jax.tree_util.tree_map(sp, state.params)

        optimizer = optax.adam(**state.hypers.optimizer.__dict__)
        optim = optimizer.init(params)

        return replace(state, key=key, params=params, optim=optim)
