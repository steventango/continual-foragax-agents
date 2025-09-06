from PyExpUtils.collection.Collector import Collector
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from algorithms.BaseAgent import BaseAgent


def reconstruct_path(start: Tuple[int, int],
                     goal: Tuple[int, int],
                     parent: Dict[Tuple[int, int], Tuple[int, int]],
                     a_map: Dict[str, int],
                     aperture: int,
                     wrapped_world: bool) -> List[int]:
    """
    Reconstructs the path (as a list of actions) from start to goal using parent pointers.
    """
    path = []
    current = goal
    while current != start:
        p = parent[current]
        dr, dc = current[0] - p[0], current[1] - p[1]
        if dr == -1 or (wrapped_world and current[0] == aperture - 1 and p[0] == 0):
            action = a_map['up']
        elif dc == 1 or (wrapped_world and current[1] == 0 and p[1] == aperture - 1):
            action = a_map['right']
        elif dr == 1 or (wrapped_world and current[0] == 0 and p[0] == aperture - 1):
            action = a_map['down']
        elif dc == -1 or (wrapped_world and current[1] == aperture - 1 and p[1] == 0):
            action = a_map['left']
        else:
            raise ValueError(f"Invalid movement from {p} to {current}")
        path.append(action)
        current = p
    path.reverse()
    return path


def bfs(state: np.ndarray,
        starting_pos: Tuple[int, int],
        a_map: Dict[str, int],
        wrapped_world: bool = False,
        target_values: List[int] = [2]) -> Tuple[Optional[Tuple[int, int]], List[int]]:
    """
    Performs breadth-first search (BFS) to locate a target based on priority.
    Priority is more important than distance. If a highest-priority target
    (assumed to be target_values[0]) is encountered, its path is returned immediately.
    """
    aperture = state.shape[0]

    # If the starting position already has a target, check adjacent cells first.
    if state[starting_pos] in target_values:
        neighbors = [
            ((starting_pos[0] - 1, starting_pos[1]), a_map['up']),
            ((starting_pos[0], starting_pos[1] + 1), a_map['right']),
            ((starting_pos[0] + 1, starting_pos[1]), a_map['down']),
            ((starting_pos[0], starting_pos[1] - 1), a_map['left'])
        ]
        # Return an immediate adjacent target.
        for pos, action in neighbors:
            if (0 <= pos[0] < aperture and 0 <= pos[1] < aperture and
                    state[pos] in target_values):
                return pos, [action]
        # If no adjacent target, try an empty cell.
        for pos, action in neighbors:
            if 0 <= pos[0] < aperture and 0 <= pos[1] < aperture and state[pos] == 0:
                opposites = {
                    a_map['up']: a_map['down'],
                    a_map['right']: a_map['left'],
                    a_map['down']: a_map['up'],
                    a_map['left']: a_map['right']
                }
                return starting_pos, [opposites[action], action]
        print('Sad opening...')
        return None, []

    queue = [starting_pos]
    visited = {starting_pos}
    parent = {}
    # Dictionary to record the first found instance for each target value.
    found_goals = {tv: None for tv in target_values}

    while queue:
        current = queue.pop(0)
        if state[current] in target_values:
            tv = state[current]
            # Immediately return if we found the highest-priority target.
            if tv == target_values[0]:
                return current, reconstruct_path(starting_pos, current, parent, a_map, aperture, wrapped_world)
            if found_goals[tv] is None:
                found_goals[tv] = current

        # Explore neighbors in random order.
        directions = [
            (a_map['up'], (-1, 0)),
            (a_map['right'], (0, 1)),
            (a_map['down'], (1, 0)),
            (a_map['left'], (0, -1))
        ]
        np.random.shuffle(directions)

        for action, (dr, dc) in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if wrapped_world:
                nr %= aperture
                nc %= aperture
            elif not (0 <= nr < aperture and 0 <= nc < aperture):
                continue

            next_pos = (nr, nc)
            if next_pos in visited or state[next_pos] == 1:
                continue

            queue.append(next_pos)
            visited.add(next_pos)
            parent[next_pos] = current

    # After full search, choose the highest-priority found target.
    for tv in target_values:
        if found_goals[tv] is not None:
            return found_goals[tv], reconstruct_path(starting_pos, found_goals[tv], parent, a_map, aperture, wrapped_world)
    return None, []


def avoid_wall_random(state: np.ndarray,
                      starting_pos: Tuple[int, int],
                      a_map: Dict[str, int]) -> int:
    """
    Returns a random action that doesn't lead to a wall.
    """
    aperture = state.shape[0]
    directions = [
        (a_map['up'], (-1, 0)),
        (a_map['right'], (0, 1)),
        (a_map['down'], (1, 0)),
        (a_map['left'], (0, -1))
    ]
    np.random.shuffle(directions)
    for action, (dr, dc) in directions:
        nr, nc = starting_pos[0] + dr, starting_pos[1] + dc
        if not (0 <= nr < aperture and 0 <= nc < aperture):
            continue
        if state[(nr, nc)] != 1:  # Not a wall.
            return action
    return np.random.randint(4)


class GreedyAgent(BaseAgent):
    """
    A greedy agent that searches for rewarding objects in a grid environment.
    It uses BFS to find the best target based on a priority system (lower numbers
    are higher priority). The agent can also explore when no target is available.
    """

    def __init__(self, observations: Tuple[int, ...], actions: int,
                 params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        self.explore_strategy = params.get('explore_strategy', 'random')
        self.explore_pump = params.get('explore_pump', 1.5)
        self.action_trace_rate = params.get('action_trace_rate', 0.9999999)
        self.env_params = params.get('env_info', {})

        self.observation_mode = self.env_params.get('observation_mode', 'objects')
        if self.observation_mode == 'objects':
            self.a_map = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
            self.wrapped_world = False
        else:
            self.a_map = {'up': 2, 'right': 1, 'down': 0, 'left': 3}
            self.wrapped_world = True

        # object_priority: mapping from object name to its assigned value (1 = obstacle, >1 = target)
        # object_channel: mapping from object name to its channel index in the observation
        self.object_priority: Dict[str, int] = self.env_params.get('object_priority', {})
        self.object_channel: Dict[str, int] = self.env_params.get('object_channel', {})

        # Determine valid target values (i.e. priority values > 1) sorted in ascending order (lowest means highest priority)
        self.target_values: List[int] = sorted([prio for prio in self.object_priority.values() if prio > 1])

        # Initialize state variables.
        self.current_goal = None
        self.current_path: List[int] = []
        self.current_goal_priority = None

        self.aperture = None
        self.starting_pos = None
        self.action_trace = np.zeros((actions,))
        self.last_action = None
        self.last_state = None
        self.step_count = 0

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms a multi-channel observation into a 2D state matrix.
        Each cell is assigned a value based on object_priority and object_channel.
        0 means empty, 1 means obstacle, and values > 1 denote targets.
        """
        state = np.zeros((self.aperture, self.aperture), dtype=int)

        # In world mode, assume channel 0 encodes the agent's position.
        if self.observation_mode == 'world':
            agent_pos = np.where(x[:, :, 0] == 1)
            if len(agent_pos[0]) > 0:
                self.starting_pos = (agent_pos[0][0], agent_pos[1][0])

        # Process each object type according to its channel and priority.
        # if there are negative values in x, the order of the objects is
        # reversed, so we need to reverse the order of the object_priority
        _reversed = np.any(x < 0)
        object_priority = {k: v for k, v in self.object_priority.items()}
        if _reversed:
            assert len(self.object_priority) == 2, "Only two objects are supported in reversed mode."
            # swap the priority values
            keys = list(object_priority.keys())
            cached_value = object_priority[keys[0]]
            object_priority[keys[0]] = object_priority[keys[1]]
            object_priority[keys[1]] = cached_value
        for obj, prio in object_priority.items():
            if obj in self.object_channel:
                channel = self.object_channel[obj]
                mask = x[:, :, channel] > 0 if not _reversed else x[:, :, channel] < 0
                # If a cell already has a value, choose the minimum (i.e. highest priority) in case of conflict.
                state[mask] = np.where(state[mask] == 0, prio, np.minimum(state[mask], prio))
        return state

    def start(self, x: np.ndarray) -> int:
        if self.aperture is None:
            self.aperture = x.shape[0]
            if self.starting_pos is None:
                self.starting_pos = (self.aperture // 2, self.aperture // 2)
        return self.step(0, x, {})

    def _check_for_better_goal(self, state: np.ndarray) -> bool:
        if self.current_goal is None or self.current_goal == self.starting_pos:
            return False

        if self.current_goal_priority is None and 0 <= self.current_goal[0] < state.shape[0] and 0 <= self.current_goal[1] < state.shape[1]:
            self.current_goal_priority = state[self.current_goal]
            if self.current_goal_priority not in self.target_values:
                self.current_goal_priority = max(self.target_values) if self.target_values else None

        new_goal, new_path = bfs(state, self.starting_pos, self.a_map, self.wrapped_world, self.target_values)
        if new_goal is None:
            return False

        new_priority = state[new_goal]
        if self.current_goal_priority is None or new_priority < self.current_goal_priority:
            self.current_goal = new_goal
            self.current_path = new_path
            self.current_goal_priority = new_priority
            return True
        return False

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]) -> int:
        state = self.transform(xp)

        # Find a new goal if needed.
        if self.current_goal is None or self.current_goal == self.starting_pos:
            self.current_goal, self.current_path = bfs(state, self.starting_pos, self.a_map, self.wrapped_world, self.target_values)
            if self.current_goal is not None:
                self.current_goal_priority = state[self.current_goal]
            else:
                self.current_goal_priority = None
        else:
            self._check_for_better_goal(state)

        # If no target is found, use the exploration strategy.
        if self.current_goal is None:
            if self.explore_strategy == 'random':
                action = avoid_wall_random(state, self.starting_pos, self.a_map)
            elif self.explore_strategy == 'trace':
                highest = np.argmax(self.action_trace)
                hit_wall = np.array_equal(state, self.last_state) if self.last_state is not None else False
                if highest == self.last_action or hit_wall:
                    self.action_trace[self.last_action] *= self.explore_pump
                action = (1 if self.action_trace[1] < self.action_trace[3] else 3) if highest in [0, 2] else \
                         (0 if self.action_trace[0] < self.action_trace[2] else 2)
            else:
                raise ValueError(f'Unknown explore strategy: {self.explore_strategy}')
        else:
            # Use pop(0) so that the next action in the planned path is executed.
            action = self.current_path.pop(0)
            if self.observation_mode == 'objects':
                # Adjust the goal relative to the agent's movement.
                dr, dc = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}[action]
                self.current_goal = (self.current_goal[0] + dr, self.current_goal[1] + dc)

        self.action_trace *= self.action_trace_rate
        self.action_trace[action] += 1 - self.action_trace_rate

        self.last_action = action
        self.last_state = state.copy()
        self.step_count += 1

        return action

    def print_state(self, x: np.ndarray) -> None:
        """
        Prints the current state with labels.
        """
        if self.starting_pos and (self.observation_mode == 'objects' or len(x.shape) == 2):
            display_x = x.copy()
            if len(x.shape) == 2:
                display_x[self.starting_pos] = -1  # Mark agent position.
                print("State representation:")
                print("  -1: Agent position")
                print("   0: Empty space")
                print("   1: Obstacle/wall")
                for obj, prio in self.object_priority.items():
                    if prio > 1:
                        print(f"   {prio}: {obj}")
                for row in display_x:
                    print(' '.join(str(int(col)) for col in row))
                print()
            else:
                for channel in range(x.shape[2]):
                    if channel == 0 and self.observation_mode == 'world':
                        channel_name = "Agent"
                    else:
                        channel_name = None
                        for obj, ch in self.object_channel.items():
                            if ch == channel:
                                channel_name = obj
                                break
                        if channel_name is None:
                            channel_name = f"Channel {channel}"
                    print(f'{channel_name}:')
                    for row in x[:, :, channel]:
                        print(' '.join(str(int(col)) for col in row))
                    print()
                print()
