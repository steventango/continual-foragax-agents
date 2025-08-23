from typing import Any

import jax
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
        self.key, env_step_key = jax.random.split(self.key)
        obs, self.state, reward, done, info = self.env.step(
            env_step_key, self.state, action
        )
        return obs, reward, done, done, info
