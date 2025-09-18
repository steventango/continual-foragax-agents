from functools import partial

import jax
import jax.numpy as jnp

import utils.chex as cxu
from utils.rlglue import BaseEnvironment


@cxu.dataclass
class EnvState:
    state: jax.Array
    key: jax.Array


class Debug(BaseEnvironment):
    def __init__(self, seed: int, **env_params):
        self.seed = seed
        self.state = EnvState(
            state=jnp.zeros((), dtype=jnp.float32), key=jax.random.key(seed)
        )

    def start(self) -> jax.Array:
        self.state, obs = self._start(self.state)
        return obs

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: EnvState) -> tuple[EnvState, jax.Array]:
        state.key, env_reset_key = jax.random.split(state.key)
        obs, state.state = state.state, state.state + 1
        return state, obs

    def step(self, action: jax.Array):
        self.state, (obs, reward, done, done, info) = self._step(
            self.state,
            action,
        )
        return (obs, reward, done, done, info)

    @partial(jax.jit, static_argnums=0)
    def _step(self, state: EnvState, action: jax.Array):
        state.key, env_step_key = jax.random.split(state.key)
        obs, state.state, reward, done, info = (
            state.state,
            state.state + 1,
            state.state + 0.1 * action,
            False,
            {},
        )
        return state, (obs, reward, done, done, info)
