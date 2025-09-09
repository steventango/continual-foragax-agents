from functools import partial
from typing import Dict, Tuple

import jax
import jax.debug
import jax.numpy as jnp
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent


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

    @partial(jax.jit, static_argnums=0)
    def act(
        self,
        state: AgentState,
        obs: jax.Array,
    ) -> tuple[AgentState, jax.Array]:
        jax.debug.print("obs: {obs}", obs=obs)
        height, width, num_channels = obs.shape
        center_y, center_x = height // 2, width // 2

        # Create priority map: higher values = higher priority
        priority_map = jnp.zeros((height, width))
        jax.debug.print(
            "initial priority_map: {priority_map}", priority_map=priority_map
        )

        # For each channel (object type), add its priority to locations where that object exists
        for channel in range(num_channels):
            channel_priority = self.channel_priorities.get(channel, 0)
            priority_map += obs[:, :, channel] * channel_priority

        jax.debug.print("final priority_map: {priority_map}", priority_map=priority_map)

        # Find the best target using BFS-like shortest path search
        best_action = self._find_best_action_bfs(
            priority_map, center_y, center_x, height, width
        )
        jax.debug.print("best_action: {best_action}", best_action=best_action)

        # If no good action found, fall back to random
        state.key, sample_key = jax.random.split(state.key)
        random_action = jax.random.choice(sample_key, self.actions)
        action = jax.lax.select(best_action >= 0, best_action, random_action)
        jax.debug.print("action: {action}", action=action)

        return state, action

    def _find_best_action_bfs(
        self,
        priority_map: jax.Array,
        center_y: int,
        center_x: int,
        height: int,
        width: int,
    ) -> jax.Array:
        """Find the best action using BFS-like shortest path search to reach highest priority object."""
        jax.debug.print("priority_map: {priority_map}", priority_map=priority_map)

        # Directions: UP, RIGHT, DOWN, LEFT (matching Actions enum)
        directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        # Initialize distance map (inf = unvisited, finite = distance from start)
        distance_map = jnp.full((height, width), jnp.inf)
        distance_map = distance_map.at[center_y, center_x].set(0)
        jax.debug.print(
            "initial distance_map: {distance_map}", distance_map=distance_map
        )

        # Initialize action map (tracks which action leads to shortest path)
        action_map = jnp.full((height, width), -1, dtype=jnp.int32)
        jax.debug.print("initial action_map: {action_map}", action_map=action_map)

        # BFS iterations - run enough iterations to cover the entire aperture
        max_iterations = height + width  # Upper bound on shortest path length

        def bfs_iteration(iter_num, state):
            dist_map, act_map = state

            # For this iteration, find all cells at distance iter_num
            current_distance_mask = dist_map == iter_num

            # For each direction, check neighbors
            def check_direction(direction_idx, maps_state):
                dist_map_inner, act_map_inner = maps_state
                dy, dx = directions[direction_idx]

                # Calculate neighbor positions
                ys, xs = jnp.indices((height, width))
                neighbor_ys = ys + dy
                neighbor_xs = xs + dx

                # Check bounds
                valid_neighbors = (
                    (neighbor_ys >= 0)
                    & (neighbor_ys < height)
                    & (neighbor_xs >= 0)
                    & (neighbor_xs < width)
                )

                # Find cells that can reach unvisited neighbors
                can_reach_unvisited = (
                    current_distance_mask
                    & valid_neighbors
                    & (dist_map_inner[neighbor_ys, neighbor_xs] == jnp.inf)
                )

                # Update distances and actions for newly discovered cells
                new_distance = iter_num + 1

                # Update distance map
                neighbor_updates = can_reach_unvisited
                dist_map_inner = jnp.where(
                    neighbor_updates,
                    dist_map_inner.at[neighbor_ys, neighbor_xs].set(new_distance),
                    dist_map_inner,
                )

                # Update action map (store the action that leads to this cell)
                act_map_inner = jnp.where(
                    neighbor_updates,
                    act_map_inner.at[neighbor_ys, neighbor_xs].set(direction_idx),
                    act_map_inner,
                )

                return (dist_map_inner, act_map_inner)

            # Apply all four directions
            final_maps = jax.lax.fori_loop(0, 4, check_direction, (dist_map, act_map))
            return final_maps

        # Run BFS iterations
        final_distance_map, final_action_map = jax.lax.fori_loop(
            0, max_iterations, bfs_iteration, (distance_map, action_map)
        )
        jax.debug.print(
            "final_distance_map: {final_distance_map}",
            final_distance_map=final_distance_map,
        )
        jax.debug.print(
            "final_action_map: {final_action_map}", final_action_map=final_action_map
        )

        # Now find the best target based on priority and distance
        # Create a combined score: higher priority, lower distance is better
        # Use priority as primary criterion, distance as tiebreaker

        # Exclude agent's position and zero-priority cells
        valid_targets = (priority_map > 0) & (jnp.isfinite(final_distance_map))
        valid_targets = valid_targets.at[center_y, center_x].set(False)

        # Create a combined score for each cell
        # Higher priority = better, lower distance = better
        # Scale priority much higher than distance to prioritize by priority first
        max_distance = jnp.max(
            jnp.where(jnp.isfinite(final_distance_map), final_distance_map, 0)
        )
        priority_weight = (max_distance + 1) * 1000  # Ensure priority dominates

        combined_score = jnp.where(
            valid_targets, priority_map * priority_weight - final_distance_map, -jnp.inf
        )
        jax.debug.print(
            "combined_score: {combined_score}", combined_score=combined_score
        )

        # Find the cell with the highest combined score
        best_cell_idx = jnp.argmax(combined_score.flatten())
        best_y = best_cell_idx // width
        best_x = best_cell_idx % width

        # Get the action that leads to the best target
        best_action = final_action_map[best_y, best_x]
        jax.debug.print("best_action: {best_action}", best_action=best_action)

        # Return -1 if no valid target found
        has_valid_target = jnp.any(valid_targets)
        return jax.lax.select(has_valid_target, best_action, -1)

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
