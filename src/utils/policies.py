import jax.numpy as jnp
import numpy as np
from typing import Any, Callable, Sequence
from PyExpUtils.utils.types import NpList
from PyExpUtils.utils.random import sample
from PyExpUtils.utils.arrays import argsmax


class Policy:
    def __init__(self, probs: Callable[[Any], NpList], rng: np.random.Generator):
        self.probs = probs
        self.random = rng

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return sample(np.asarray(action_probabilities), rng=self.random)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]


def fromStateArray(probs: Sequence[NpList], rng: np.random.Generator):
    return Policy(lambda s: probs[s], rng)


def fromActionArray(probs: NpList, rng: np.random.Generator):
    return Policy(lambda s: probs, rng)


def createEGreedy(
    get_values: Callable[[Any], np.ndarray],
    actions: int,
    epsilon: float,
    rng: np.random.Generator,
):
    probs = lambda state: egreedy_probabilities(get_values(state), actions, epsilon)

    return Policy(probs, rng)


def egreedy_probabilities(qs: jnp.ndarray, actions: int, epsilon: float):
    # compute the greedy policy
    pi = qs == jnp.max(qs)

    # compute a uniform random policy
    uniform = jnp.ones(actions) / actions

    # epsilon greedy is a mixture of greedy + uniform random
    return (1.0 - epsilon) * pi + epsilon * uniform
