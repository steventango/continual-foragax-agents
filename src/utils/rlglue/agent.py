from abc import abstractmethod
from functools import partial
from typing import Any, Protocol

import jax


class BaseAgent(Protocol):
    @abstractmethod
    def start(self, obs: jax.Array) -> jax.Array:
        raise NotImplementedError("Expected `start` to be implemented")

    @abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _start(self, state: Any, obs: jax.Array) -> tuple[Any, jax.Array]:
        raise NotImplementedError("Expected `_start` to be implemented")

    @abstractmethod
    def step(self, reward: jax.Array, obs: jax.Array, extra: dict[str, Any]) -> int:
        raise NotImplementedError("Expected `step` to be implemented")

    @abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: Any,
        reward: jax.Array,
        obs: jax.Array,
        extra: dict[str, jax.Array],
    ) -> tuple[Any, jax.Array]:
        raise NotImplementedError("Expected `_step` to be implemented")

    @abstractmethod
    def end(self, reward: jax.Array, extra: dict[str, Any]) -> None:
        raise NotImplementedError("Expected `end` to be implemented")

    @abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _end(
        self, state: Any, reward: jax.Array, extra: dict[str, jax.Array]
    ) -> Any:
        raise NotImplementedError("Expected `_end` to be implemented")
