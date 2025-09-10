from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Queue:
    """A fixed-size queue implemented with JAX arrays."""

    buffer: jax.Array
    head: int
    tail: int
    size: int
    max_size: int

    @classmethod
    def create(cls, max_size: int, dtype: Any = jnp.float32, item_shape: tuple = ()):
        """Creates an empty queue."""
        buffer = jnp.zeros((max_size,) + item_shape, dtype=dtype)
        return cls(buffer=buffer, head=0, tail=0, size=0, max_size=max_size)


def enqueue(queue: Queue, item: Any) -> Queue:
    """Adds an item to the end of the queue if not full."""

    def _enqueue_fn(q: Queue):
        new_buffer = q.buffer.at[q.tail].set(item)
        new_tail = (q.tail + 1) % q.max_size
        new_size = q.size + 1
        return q.replace(buffer=new_buffer, tail=new_tail, size=new_size)

    def _overflow_fn(q: Queue):
        return q

    return jax.lax.cond(queue.size < queue.max_size, _enqueue_fn, _overflow_fn, queue)


def dequeue(queue: Queue) -> Tuple[Queue, Any]:
    """Removes an item from the front of the queue."""

    def _dequeue_fn(q: Queue) -> Tuple[Queue, Any]:
        item = q.buffer[q.head]
        new_head = (q.head + 1) % q.max_size
        new_size = q.size - 1
        # The dequeued item remains in the buffer but is inaccessible.
        # It will be overwritten if the queue wraps around.
        return q.replace(head=new_head, size=new_size), item

    def _underflow_fn(q: Queue) -> Tuple[Queue, Any]:
        # Return a default value (zeros) and the unchanged queue.
        item_shape = q.buffer.shape[1:]
        item = jnp.zeros(item_shape, dtype=q.buffer.dtype)
        return q, item

    return jax.lax.cond(queue.size > 0, _dequeue_fn, _underflow_fn, queue)
