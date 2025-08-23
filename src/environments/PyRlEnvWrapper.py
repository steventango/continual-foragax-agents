import jax
import jax.numpy as jnp
from rlglue import BaseEnvironment


class PyRlEnvWrapper(BaseEnvironment):
    def __init__(self, env):
        self.env = env
        pass

    def start(self):
        return jnp.asarray(self.env.start())

    def step(self, action: jax.Array):
        action = action.item()
        obs, reward, terminal, truncated, info = self.env.step(action)
        return (
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.float32(reward),
            terminal,
            truncated,
            info,
        )
