from functools import partial

import jax

from utils.rlglue import BaseEnvironment


class MCTSEnvWrapper(BaseEnvironment):
    def __init__(self, env: BaseEnvironment):
        super().__init__()
        self._env = env

    # pass getattr to the wrapped env
    def __getattr__(self, name):
        return getattr(self._env, name)

    def start(self) -> jax.Array:
        self.state, obs = self._start(self.state)
        return obs

    @partial(jax.jit, static_argnums=0)
    def _start(self, state):
        state, obs = self._env._start(state)
        # Return tuple of (obs, state) so MCTS can use both
        obs_with_state = (obs, state)
        return state, obs_with_state

    def step(self, action: jax.Array):
        self.state, (obs, reward, terminated, truncated, info) = self._step(
            self.state,
            action,
        )
        return (obs, reward, terminated, truncated, info)

    @partial(jax.jit, static_argnums=0)
    def _step(self, state, action: jax.Array):
        state, (obs, reward, terminated, truncated, info) = self._env._step(state, action)
        # Return tuple of (obs, state) so MCTS can use both
        obs_with_state = (obs, state)
        return state, (obs_with_state, reward, terminated, truncated, info)
