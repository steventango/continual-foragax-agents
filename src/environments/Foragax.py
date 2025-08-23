from typing import Any
from functools import partial
import jax
from foragax.env import EnvState
from foragax.registry import make
from rlglue import BaseEnvironment


class Foragax(BaseEnvironment):
    def __init__(self, seed: int, **env_params):
        self.env = make(**env_params)
        self.seed = seed
        self.state = None
        self.key = jax.random.PRNGKey(seed)

    def start(self) -> Any:
        self.key, env_reset_key = jax.random.split(self.key)
        obs, self.state = self.env.reset(env_reset_key)
        return obs

    def step(self, action: int):
        (self.state, self.key), (obs, reward, done, done, info) = self._step(
            self.state, action, self.key
        )
        return (obs, reward, done, done, info)

    @partial(jax.jit, static_argnums=0)
    def _step(self, state: EnvState, action: int, key: jax.Array):
        key, env_step_key = jax.random.split(key)
        obs, state, reward, done, info = self.env.step(env_step_key, state, action)
        return (state, key), (obs, reward, done, done, info)
