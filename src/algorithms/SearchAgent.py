from dataclasses import replace
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent
from utils.queue import Queue, dequeue, enqueue


@cxu.dataclass
class AgentState:
    key: jax.Array


class SearchAgent(BaseAgent):
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
        self.channel_priorities = params.get("channel_priorities", {})
        self.channel_priorities = {
            int(k): v for k, v in self.channel_priorities.items()
        }
        max_channel = (
            max(self.channel_priorities.keys()) if self.channel_priorities else 0
        )
        self.priorities_array = jnp.array(
            [self.channel_priorities.get(i, 0) for i in range(max_channel + 1)]
        )
        self.temperature_prioritization = params.get(
            "temperature_prioritization", False
        )
        if len(self.channel_priorities):
            self.max_priority = max(self.channel_priorities.values())
        else:
            self.max_priority = observations[-1]
        self.mode = params.get("mode", "aperture")
        self.nowrap = params.get("nowrap", False)
        # new option: use temperatures from extra to determine priorities
        # priorities:
        # higher values = higher priority
        # zero = ignore
        # negative values = obstacles; avoid
        self.directions = jnp.array(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )

    def _get_world_position(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Get agent position from the last channel (world mode)."""
        agent_channel = obs[:, :, -1]
        agent_pos = jnp.argwhere(agent_channel > 0, size=1)[0]
        return agent_pos[0], agent_pos[1]

    @partial(jax.jit, static_argnums=0)
    def act(
        self,
        state: AgentState,
        obs: jax.Array,
        extra: Optional[Dict[str, jax.Array]] = None,
    ) -> tuple[AgentState, jax.Array]:
        height, width, num_channels = obs.shape

        if self.mode == "world":
            center_y, center_x = self._get_world_position(obs)
            obs = obs[:, :, :-1]
            num_channels -= 1
        else:
            center_y, center_x = height // 2, width // 2

        # Create priority map: higher values = higher priority
        priority_map = jnp.zeros((height, width))

        if self.temperature_prioritization and extra is not None:
            # Skip temperature of empty object
            temps = extra["temperatures"][1:]
            # Assign descending integer priorities (highest temp -> largest priority)
            order = jnp.argsort(temps, descending=True)
            priorities = jnp.zeros(num_channels, dtype=jnp.int32)
            priorities = priorities.at[order].set(jnp.arange(num_channels, 0, -1))
            # If negative temperature, set priority to -1 (obstacle)
            priorities = jnp.where(temps < 0, -1, priorities)
        else:
            priorities = self.priorities_array[:num_channels]

        # For each channel (object type), add its priority to locations where that object exists
        for channel in range(num_channels):
            channel_priority = priorities[channel]
            priority_map += obs[:, :, channel] * channel_priority

        # Find the best target using BFS
        highest_priority = self.max_priority

        def cond_fun(carry):
            _, current_priority, best_action, _ = carry
            return (current_priority > 0) & (best_action < 0)

        def body_fun(carry):
            priority_map, current_priority, best_action, key = carry
            priority_map_masked = jax.lax.select(
                (priority_map < 0) | (priority_map == current_priority),
                priority_map,
                jnp.zeros_like(priority_map),
            )
            key, best_action = self.bfs(
                key, priority_map_masked, center_y, center_x, height, width
            )
            current_priority -= 1
            return priority_map, current_priority, best_action, key

        # do BFS until we find a good action or run out of priorities
        *_, best_action, key = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (priority_map, highest_priority, -1, state.key),
        )
        state = replace(state, key=key)

        # If no good action found, fall back to random valid action
        next_positions = jnp.array([center_y, center_x]) + self.directions
        if not self.nowrap:
            next_positions = next_positions % jnp.array([height, width])
        next_priorities = priority_map[next_positions[:, 0], next_positions[:, 1]]
        if self.nowrap:
            in_bounds = (
                (next_positions[:, 0] >= 0)
                & (next_positions[:, 0] < height)
                & (next_positions[:, 1] >= 0)
                & (next_positions[:, 1] < width)
            )
            next_priorities = jnp.where(in_bounds, next_priorities, -1)
        valid_mask = next_priorities >= 0
        probs = jnp.where(valid_mask, 1.0, 0.0)
        key, sample_key = jax.random.split(state.key)
        state = replace(state, key=key)
        random_action = jax.lax.cond(
            jnp.sum(probs) > 0,
            lambda: jax.random.choice(
                sample_key, jnp.arange(4), p=probs / jnp.sum(probs)
            ),
            lambda: jax.random.choice(sample_key, jnp.arange(4)),
        )
        action = jax.lax.select(best_action >= 0, best_action, random_action)

        return state, action

    def bfs(
        self,
        key: jax.Array,
        priority_map: jax.Array,
        start_y: jax.Array,
        start_x: jax.Array,
        height: int,
        width: int,
    ) -> tuple[jax.Array, int]:
        queue = Queue.create(max_size=height * width, dtype=jnp.int32, item_shape=(2,))
        start = jnp.array([start_y, start_x])
        queue = enqueue(queue, start)
        visited = jnp.zeros((height, width), dtype=jnp.bool_)
        visited = visited.at[start_y, start_x].set(True)
        actions = jnp.full((height, width), -1, dtype=jnp.int32)

        def cond_fun(carry):
            queue, *_, target, _ = carry
            return (queue.size > 0) & (target.sum() < 0)

        def body_fun(carry):
            queue, visited, best_actions, _, key = carry
            queue, node = dequeue(queue)
            y, x = node

            def found_target_branch(carry):
                queue, visited, best_actions, _, key = carry
                return queue, visited, best_actions, node, key

            def inner_fun(carry):
                queue, visited, best_actions, _, key = carry

                new_key, sample_key = jax.random.split(key)
                shuffled_indices = jax.random.permutation(
                    sample_key, jnp.arange(self.directions.shape[0])
                )

                def loop_body(i, loop_carry):
                    queue, visited, best_actions = loop_carry
                    direction_idx = shuffled_indices[i]
                    neighbor = node + self.directions[direction_idx]
                    if self.nowrap:
                        ny, nx = neighbor
                        in_bounds = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
                        is_valid = (
                            in_bounds & (priority_map[ny, nx] >= 0) & (~visited[ny, nx])
                        )
                    else:
                        neighbor = neighbor % jnp.array([height, width])
                        ny, nx = neighbor
                        is_valid = (
                            (priority_map[ny, nx] >= 0)  # not an obstacle
                            & (~visited[ny, nx])  # not visited
                        )

                    def enqueue_fn(carry):
                        queue, visited, best_actions = carry
                        return (
                            enqueue(queue, neighbor),
                            visited.at[ny, nx].set(True),
                            best_actions.at[ny, nx].set(direction_idx),
                        )

                    queue, visited, best_actions = jax.lax.cond(
                        is_valid,
                        enqueue_fn,
                        lambda carry: carry,
                        operand=(queue, visited, best_actions),
                    )
                    return queue, visited, best_actions

                queue, visited, best_actions = jax.lax.fori_loop(
                    0,
                    self.directions.shape[0],
                    loop_body,
                    (queue, visited, best_actions),
                    unroll=True,
                )
                return queue, visited, best_actions, jnp.array([-1, -1]), new_key

            return jax.lax.cond(
                priority_map[y, x] > 0,
                found_target_branch,
                inner_fun,
                (queue, visited, best_actions, jnp.array([-1, -1]), key),
            )

        *_, actions, target, key = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (queue, visited, actions, jnp.array([-1, -1]), key),
        )

        best_action = jax.lax.cond(
            target.sum() > 0,
            lambda _: self.backtrack(target, start, actions),
            lambda _: -1,
            operand=None,
        )
        return key, best_action

    def backtrack(
        self,
        target: jax.Array,
        start: jax.Array,
        actions: jax.Array,
    ):
        def cond_fun(carry):
            current, _ = carry
            return (current != start).any()

        def body_fun(carry):
            current, _ = carry
            action = actions[current[0], current[1]]
            # Move to the parent node
            prev = current - self.directions[action]
            # Wrap the previous position
            if self.nowrap:
                prev_wrapped = prev
            else:
                prev_wrapped = prev % jnp.array([actions.shape[0], actions.shape[1]])
            # The action we want is the one that led from the parent to the current node
            return prev_wrapped, action

        initial_carry = (target, -1)

        _, first_action = jax.lax.while_loop(cond_fun, body_fun, initial_carry)

        return first_action

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, obs: jax.Array):
        self.state, a = self._start(self.state, obs)
        return a

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: jax.Array):
        return self.act(state, obs)

    def step(self, reward: jax.Array, obs: jax.Array, extra: Dict[str, jax.Array]):
        self.state, a = self._step(self.state, reward, obs, extra)
        return a

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: jax.Array,
        extra: Dict[str, jax.Array],
    ):
        return self.act(state, obs, extra)

    def end(self, reward: jax.Array, extra: Dict[str, jax.Array]):
        self.state = self._end(self.state, reward, extra)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
        return state
