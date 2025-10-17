from dataclasses import replace
from functools import partial
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import mctx
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent
from environments.MCTSEnvWrapper import MCTSEnvWrapper


@cxu.dataclass
class AgentState:
    key: jax.Array


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.state = AgentState(
            key=self.key,
        )
        self.max_depth = params.get("max_depth")
        self.num_simulations = params["num_simulations"]
        self.dirichlet_fraction = float(params.get("dirichlet_fraction", 0.25))
        self.dirichlet_alpha = float(params.get("dirichlet_alpha", 0.3))
        self.pb_c_init = float(params.get("pb_c_init", 1.25))
        self.pb_c_base = float(params.get("pb_c_base", 19652))
        self.temperature = float(params.get("temperature", 1.0))
        self.env: MCTSEnvWrapper = None  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def act(
        self,
        state: AgentState,
        obs,
    ) -> tuple[AgentState, jax.Array]:
        key, subkey = jax.random.split(state.key)
        state = replace(state, key=key)

        policy_output = self.run_mcts(subkey, obs, self.env._step)
        # Remove batch dimension from action (mctx returns batched actions)
        action = policy_output.action[0]

        return state, action

    @partial(jax.jit, static_argnums=(0, 3))
    def run_mcts(
        self, rng_key: chex.PRNGKey, env_state, env_step_fn
    ) -> mctx.PolicyOutput:
        def root_fn(
            env_state: chex.Array, rng_key: chex.PRNGKey
        ) -> mctx.RootFnOutput:
            # Simple uniform policy for priors
            prior_logits = jnp.zeros(self.actions)

            # Value is estimated with a random rollout
            value = self.value_function(env_state, rng_key, env_step_fn)

            return mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=env_state,
            )

        def recurrent_fn(params, rng_key, action, embedding):
            env_state = embedding

            # Note: MCTSEnvWrapper._step returns (next_env_state, (obs, reward, done, done, info))
            next_env_state, (_, reward, _, _, _) = env_step_fn(env_state, action)

            # For continuing environments, done is always False, so we always use discount of 1.0
            discount = jnp.ones_like(reward).astype(jnp.float32)

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward.astype(jnp.float32),
                discount=discount,
                prior_logits=jnp.zeros(self.actions),
                value=self.value_function(next_env_state, rng_key, env_step_fn).astype(jnp.float32),
            )

            return recurrent_fn_output, next_env_state

        # Create batch dimension for mctx
        batch_size = 1
        key1, key2 = jax.random.split(rng_key)

        policy_output = mctx.muzero_policy(
            params={},
            rng_key=key1,
            # Create a batch of environments
            root=jax.vmap(root_fn, (None, 0))(env_state, jax.random.split(key2, batch_size)),
            # Automatically vectorize the recurrent_fn exactly like Connect 4
            recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
            num_simulations=self.num_simulations,
            max_depth=self.max_depth,
            dirichlet_fraction=self.dirichlet_fraction,
            dirichlet_alpha=self.dirichlet_alpha,
            pb_c_init=self.pb_c_init,
            pb_c_base=self.pb_c_base,
            temperature=self.temperature,
        )
        return policy_output

    def value_function(
        self, env_state, rng_key: chex.PRNGKey, env_step_fn
    ) -> float:
        """Estimates the value of a state with a random rollout.

        For continuing environments where done is always False, we run a fixed number
        of steps equal to max_depth (or a default if max_depth is None).
        """
        def step_fun(carry, _):
            key, env_state, total_reward = carry
            key, subkey = jax.random.split(key)
            action = jax.random.randint(
                subkey, shape=(), minval=0, maxval=self.actions
            )

            next_env_state, (_, reward, _, _, _) = env_step_fn(env_state, action)

            total_reward += reward
            return (key, next_env_state, total_reward), None

        (_, _, total_reward), _ = jax.lax.scan(
            step_fun, (rng_key, env_state, 0.0), None, length=self.max_depth
        )

        return total_reward

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, obs: jax.Array):
        self.state, a = self._start(self.state, obs)
        return a.item()

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: jax.Array):
        return self.act(state, obs)

    def step(self, reward: jax.Array, obs: jax.Array, extra: Dict):
        self.state, a = self._step(self.state, reward, obs, extra)
        return a.item()

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: jax.Array,
        extra: Dict[str, jax.Array],
    ):
        return self.act(state, obs)

    def end(self, reward: jax.Array, extra: Dict):
        self.state = self._end(self.state, reward, extra)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict):
        return state
