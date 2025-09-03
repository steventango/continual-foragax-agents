from typing import Dict, Tuple

import jax
import numpy as np
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from utils.rlglue import BaseAgent as Base


@cxu.dataclass
class Hypers:
    gamma: float


@cxu.dataclass
class AgentState:
    hypers: Hypers


class BaseAgent(Base):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        self.observations = observations
        self.actions = actions
        self.params = params
        self.collector = collector

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.key = jax.random.key(seed)

        self.gamma = params.get("gamma", 1)
        self.n_step = params.get("n_step", 1)

        self.state = AgentState(
            hypers=Hypers(gamma=self.gamma),
        )

    def cleanup(self): ...

    # -------------------
    # -- Checkpointing --
    # -------------------
    def __getstate__(self):
        return {
            "__args": (
                self.observations,
                self.actions,
                self.params,
                self.collector,
                self.seed,
            ),
            "rng": self.rng,
        }

    def __setstate__(self, state):
        self.__init__(*state["__args"])
        self.rng = state["rng"]
