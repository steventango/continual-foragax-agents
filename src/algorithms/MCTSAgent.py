from dataclasses import replace
from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.debug
import jax.numpy as jnp
import mctx
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent
from algorithms.SearchAgent import SearchAgent
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

        # Initialize oracle search agent for rollouts
        search_params = params.get("search_params", {})
        self.search_agent = SearchAgent(
            observations=observations,
            actions=actions,
            params=search_params,
            collector=collector,
            seed=seed,
        )

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
        jax.debug.print("DOWN RIGHT UP LEFT")
        jax.debug.print("MCTS selected action: {action}", action=action)
        jax.debug.print(
            "MCTS action weights: {weights}", weights=policy_output.action_weights
        )
        search_tree = policy_output.search_tree
        summary = search_tree.summary()
        jax.debug.print("MCTS visit counts: {counts}", counts=summary.visit_counts)
        jax.debug.print("MCTS visit probs: {probs}", probs=summary.visit_probs)
        jax.debug.print("MCTS root value: {value}", value=summary.value)
        jax.debug.print("MCTS root Q-values: {qvalues}", qvalues=summary.qvalues)

        return state, action

    @partial(jax.jit, static_argnums=(0, 3))
    def run_mcts(
        self, rng_key: chex.PRNGKey, env_state, env_step_fn
    ) -> mctx.PolicyOutput:

        def prior_fn(obs):
            height, width, num_channels = obs.shape

            # Get agent position
            if self.search_agent.mode == "world":
                center_y, center_x = self.search_agent._get_world_position(obs)
                obs_for_priority = obs[:, :, :-1]
                num_channels_for_priority = num_channels - 1
            else:
                center_y, center_x = height // 2, width // 2
                obs_for_priority = obs
                num_channels_for_priority = num_channels

            # Create priority map to identify obstacles
            priority_map = jnp.zeros((height, width))
            priorities = self.search_agent.priorities_array[:num_channels_for_priority]
            for channel in range(num_channels_for_priority):
                channel_priority = priorities[channel]
                priority_map += obs_for_priority[:, :, channel] * channel_priority

            # Get valid actions mask
            valid_actions = self.search_agent.get_valid_actions(
                obs, priority_map, center_y, center_x
            )

            # Set prior logits: very negative for invalid actions, zero for valid ones
            prior_logits = jnp.where(valid_actions, 0.0, -100.0)
            return prior_logits

        def root_fn(
            env_state: chex.Array, rng_key: chex.PRNGKey
        ) -> mctx.RootFnOutput:
            obs, state = env_state
            prior_logits = prior_fn(obs)

            # Value is estimated with a random rollout
            value = self.value_function(env_state, rng_key, env_step_fn)

            jax.debug.print("MCTS prior logits: {logits}", logits=prior_logits)

            return mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=env_state,
            )


        def recurrent_fn(params, rng_key, action, embedding):
            # embedding is (obs, state) tuple from MCTSEnvWrapper
            env_state_tuple = embedding
            obs, state = env_state_tuple

            # Note: MCTSEnvWrapper._step returns (next_state, ((next_obs, next_state), reward, done, done, info))
            # We pass only the state part, not the tuple
            next_state, (next_env_state_tuple, reward, _, _, _) = env_step_fn(
                state, action
            )

            # For continuing environments, done is always False, so we always use discount of 1.0
            discount = jnp.ones_like(reward).astype(jnp.float32) * self.gamma

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward.astype(jnp.float32),
                discount=discount,
                prior_logits=prior_fn(obs),
                value=self.value_function(
                    next_env_state_tuple, rng_key, env_step_fn
                ).astype(jnp.float32),
            )

            return recurrent_fn_output, next_env_state_tuple

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
        """Estimates the value of a state using oracle search policy rollout.

        For continuing environments where done is always False, we run a fixed number
        of steps equal to max_depth using the SearchAgent's policy.
        """
        # Initialize search agent state
        search_state = self.search_agent.state
        search_state = replace(search_state, key=rng_key)

        def cond_fun(val):
            _, _, _, _, reward = val
            return reward == 0

        def body_fun(val):
            (
                search_state,
                env_state_tuple,
                discounted_return,
                discount,
                _,
            ) = val
            obs, state = env_state_tuple

            # Use SearchAgent to select action
            search_state, action = self.search_agent.act(search_state, obs, extra=None)

            # Step environment
            _, (next_env_state_tuple, reward, _, _, _) = env_step_fn(state, action)

            # Accumulate discounted reward
            discounted_return += discount * reward
            # Update discount factor for next step
            next_discount = discount * self.gamma

            return (
                search_state,
                next_env_state_tuple,
                discounted_return,
                next_discount,
                reward,
            )

        init_val = (search_state, env_state, 0.0, 1.0, 0.0)
        final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
        discounted_return = final_val[2]

        return discounted_return

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
        jax.debug.print("MCTS received reward: {reward}", reward=reward)
        return self.act(state, obs)

    def end(self, reward: jax.Array, extra: Dict):
        self.state = self._end(self.state, reward, extra)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict):
        return state
