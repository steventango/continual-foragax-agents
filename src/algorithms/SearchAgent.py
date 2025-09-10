from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from ml_instrumentation.Collector import Collector
import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent
from utils.queue import Queue, dequeue, enqueue


DIRECTIONS = jnp.array(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ]
)

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
        self.channel_priorities = params["channel_priorities"]
        self.channel_priorities = {
            int(k): v for k, v in self.channel_priorities.items()
        }
        self.max_priority = max(self.channel_priorities.values())
        # priorities:
        # higher values = higher priority
        # zero = ignore
        # negative values = obstacles; avoid

    @partial(jax.jit, static_argnums=0)
    def act(
        self,
        state: AgentState,
        obs: jax.Array,
    ) -> tuple[AgentState, jax.Array]:
        height, width, num_channels = obs.shape
        center_y, center_x = height // 2, width // 2

        # Create priority map: higher values = higher priority
        priority_map = jnp.zeros((height, width))

        # For each channel (object type), add its priority to locations where that object exists
        for channel in range(num_channels):
            channel_priority = self.channel_priorities.get(channel, 0)
            priority_map += obs[:, :, channel] * channel_priority

        # Find the best target using BFS
        highest_priority = self.max_priority

        def cond_fun(carry):
            _, current_priority, best_action = carry
            return (current_priority > 0) & (best_action < 0)

        def body_fun(carry):
            priority_map, current_priority, best_action = carry
            priority_map_masked = jax.lax.select(
                (priority_map < 0) | (priority_map == current_priority),
                priority_map,
                jnp.zeros_like(priority_map)
            )
            best_action = self.bfs(
                priority_map_masked, center_y, center_x, height, width
            )
            current_priority -= 1
            return priority_map, current_priority, best_action

        # do BFS until we find a good action or run out of priorities
        priority_map, _, best_action = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (priority_map, highest_priority, -1),
        )

        # If no good action found, fall back to random
        state.key, sample_key = jax.random.split(state.key)
        random_action = jax.random.choice(sample_key, self.actions)
        action = jax.lax.select(best_action >= 0, best_action, random_action)

        return state, action

    def bfs(
        self,
        priority_map: jax.Array,
        start_y: int,
        start_x: int,
        height: int,
        width: int,
    ) -> int:
        queue = Queue.create(max_size=height * width, dtype=jnp.int32, item_shape=(2,))
        start = jnp.array([start_y, start_x])
        queue = enqueue(queue, start)
        visited = jnp.zeros((height, width), dtype=jnp.bool_)
        visited = visited.at[start_y, start_x].set(True)
        actions = jnp.full((height, width), -1, dtype=jnp.int32)

        def cond_fun(carry):
            queue, *_, target = carry
            return (queue.size > 0) & (target.sum() < 0)

        def body_fun(carry):
            queue, visited, best_actions, _ = carry
            queue, node = dequeue(queue)
            y, x = node

            def found_target_branch(carry):
                queue, visited, best_actions, _ = carry
                return queue, visited, best_actions, node

            def inner_fun(carry):
                queue, visited, best_actions, _ = carry
                #  9          for all edges from v to w in G.adjacentEdges(v) do
                for i in range(DIRECTIONS.shape[0]):
                    neighbor = node + DIRECTIONS[i]
                    ny, nx = neighbor
                    is_valid = (
                        (0 <= ny)
                        & (ny < height)
                        & (0 <= nx)
                        & (nx < width)
                        & (priority_map[ny, nx] >= 0)  # not an obstacle
                        & (~visited[ny, nx])  # not visited
                    )
                    # 10              if w is not labeled as explored then
                    # 11                  label w as explored
                    # 12                  w.parent := v
                    # 13                  Q.enqueue(w)
                    def enqueue_fn(carry):
                        queue, visited, best_actions = carry
                        return (
                            enqueue(queue, neighbor),
                            visited.at[ny, nx].set(True),
                            best_actions.at[ny, nx].set(i),
                        )

                    queue, visited, best_actions = jax.lax.cond(
                        is_valid,
                        enqueue_fn,
                        lambda carry: carry,
                        operand=(queue, visited, best_actions),
                    )
                return queue, visited, best_actions, jnp.array([-1, -1])

            return jax.lax.cond(
                priority_map[y, x] > 0,
                found_target_branch,
                inner_fun,
                (queue, visited, best_actions, jnp.array([-1, -1])),
            )

        queue, visited, actions, target = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (queue, visited, actions, jnp.array([-1, -1])),
        )

        best_action = jax.lax.cond(
            target.sum() > 0,
            lambda _: self.backtrack(
                target, start, actions
            ),
            lambda _: -1,
            operand=None,
        )
        return best_action

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
            prev = current - DIRECTIONS[action]
            # The action we want is the one that led from the parent to the current node
            return prev, action

        # The initial state for the loop is the target location.
        # The third element of the carry tuple is a placeholder for the action.
        initial_carry = (target, -1)

        # After the loop, final_carry will be (start_y, start_x, first_action)
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
        return self.act(state, obs)

    def end(self, reward: jax.Array, extra: Dict[str, jax.Array]):
        self.state = self._end(self.state, reward, extra)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
        return state
