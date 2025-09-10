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
    reset_mask: Dict


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
        reset_mask = {
            "phi": {
                k: jax.tree_util.tree_map(
                    lambda _: not params["reset_head_only"] or k != "phi", v
                )
                for k, v in self.state.params["phi"].items()
            },
            "q": jax.tree_util.tree_map(lambda _: True, self.state.params["q"]),
        }

        hypers = Hypers(
            **self.state.hypers.__dict__,
            reset_steps=params["reset_steps"],
            reset_mask=reset_mask,
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
        key, subkey = jax.random.split(state.key)
        reset_params = self.builder.reset(subkey)
        key, subkey = jax.random.split(key)
        reset_params["q"] = self.q_net.init(subkey, self.builder._sample_phi)

        params = jax.tree_util.tree_map(
            lambda old, new, mask: jax.lax.select(mask, new, old),
            state.params,
            reset_params,
            state.hypers.reset_mask,
        )

        optimizer = optax.adam(**state.hypers.optimizer.__dict__)
        reset_optim = optimizer.init(params)

        optim = jax.tree_util.tree_map(
            lambda old, new, mask: jax.lax.select(mask, new, old),
            state.optim,
            reset_optim,
            state.hypers.reset_mask,
        )

        return replace(state, key=key, params=params, optim=optim)
