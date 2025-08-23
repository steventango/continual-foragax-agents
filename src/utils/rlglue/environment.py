from abc import abstractmethod
from functools import partial
from typing import Any, Protocol

import jax


class BaseEnvironment(Protocol):
    @abstractmethod
    def start(self) -> Any:
        raise NotImplementedError("Expected `start` to be implemented")

    @abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _start(self, key: jax.Array) -> tuple[Any, jax.Array]:
        raise NotImplementedError("Expected `_start` to be implemented")

    @abstractmethod
    def step(self, action: jax.Array) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError("Expected `step` to be implemented")

    @abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _step(
        self, state: Any, action: jax.Array
    ) -> tuple[
        Any,
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]],
    ]:
        raise NotImplementedError("Expected `_step` to be implemented")
