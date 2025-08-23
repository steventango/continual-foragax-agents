import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
import numpy as np
import utils.chex as cxu

from abc import abstractmethod
from functools import partial
from typing import Any, Dict, Tuple
from ml_instrumentation.Collector import Collector

from algorithms.BaseAgent import BaseAgent
from representations.networks import NetworkBuilder
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities


@cxu.dataclass
class AgentState:
    params: Any
    optim: optax.OptState
    buffer_state: Any


@checkpointable(("buffer", "steps", "state", "updates"))
class NNAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        # ------------------------------
        # -- Configuration Parameters --
        # ------------------------------
        self.rep_params: Dict = params["representation"]
        self.optimizer_params: Dict = params["optimizer"]

        self.epsilon = params["epsilon"]
        self.reward_clip = params.get("reward_clip", 0)

        # ---------------------
        # -- NN Architecture --
        # ---------------------
        builder = NetworkBuilder(observations, self.rep_params, seed)
        self._build_heads(builder)
        self.phi = builder.getFeatureFunction()
        net_params = builder.getParams()

        # ---------------
        # -- Optimizer --
        # ---------------
        self.optimizer = optax.adam(
            self.optimizer_params["alpha"],
            self.optimizer_params["beta1"],
            self.optimizer_params["beta2"],
        )
        opt_state = self.optimizer.init(net_params)

        # ------------------
        # -- Data ingress --
        # ------------------
        self.buffer_size = params["buffer_size"]
        self.batch_size = params["batch"]
        self.buffer_min_size = params.get("buffer_min_size", self.batch_size)
        self.update_freq = params.get("update_freq", 1)
        self.priority_exponent = params.get("priority_exponent", 0.0)

        self.buffer = fbx.make_prioritised_trajectory_buffer(
            max_length_time_axis=self.buffer_size,
            min_length_time_axis=self.buffer_min_size,
            sample_batch_size=self.batch_size,
            add_batch_size=1,
            sample_sequence_length=self.n_step + 1,
            period=1,
            priority_exponent=self.priority_exponent,
        )
        self.buffer = self.buffer.replace(
            init=jax.jit(self.buffer.init),
            add=jax.jit(self.buffer.add, donate_argnums=(0,)),
            sample=jax.jit(self.buffer.sample),
            can_sample=jax.jit(self.buffer.can_sample),
            set_priorities=jax.jit(self.buffer.set_priorities, donate_argnums=(0,)),
        )

        dummy_timestep = {
            "x": jnp.zeros(self.observations),
            "a": jnp.int32(0),
            "r": jnp.float32(0),
            "gamma": jnp.float32(0),
        }
        buffer_state = self.buffer.init(dummy_timestep)
        self.last_timestep = dummy_timestep

        # --------------------------
        # -- Stateful information --
        # --------------------------
        self.state = AgentState(
            params=net_params,
            optim=opt_state,
            buffer_state=buffer_state,
        )

        self.steps = 0
        self.updates = 0

    # ------------------------
    # -- NN agent interface --
    # ------------------------

    @abstractmethod
    def _build_heads(self, builder: NetworkBuilder) -> None: ...

    @abstractmethod
    def _values(self, state: Any, x: jax.Array) -> jax.Array: ...

    @abstractmethod
    def update(self) -> None: ...

    def policy(self, obs: jax.Array) -> jax.Array:
        q = self.values(obs)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi

    @partial(jax.jit, static_argnums=0)
    def _policy(self, state: Any, obs: jax.Array) -> jax.Array:
        obs = jnp.expand_dims(obs, 0)
        q = self._values(state, obs)[0]
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi

    @partial(jax.jit, static_argnums=0)
    def act(
        self, state: Any, obs: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        pi = self._policy(state, obs)
        key, sample_key = jax.random.split(key)
        a = jax.random.choice(sample_key, self.actions, p=pi)
        return a, key

    # --------------------------
    # -- Base agent interface --
    # --------------------------
    def values(self, x: jax.Array):
        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = jnp.expand_dims(x, 0)
            q = self._values(self.state, x)[0]

        else:
            q = self._values(self.state, x)

        return q

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, obs: jax.Array):
        a, self.key = self.act(self.state, obs, self.key)
        self.last_timestep.update(
            {
                "x": obs,
                "a": a,
            }
        )
        return a

    def step(self, reward: jax.Array, obs: jax.Array, extra: Dict[str, Any]):
        a, self.key = self.act(self.state, obs, self.key)

        # see if the problem specified a discount term
        gamma = extra.get("gamma", 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        self.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(self.gamma * gamma),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), self.last_timestep
        )
        self.state.buffer_state = self.buffer.add(
            self.state.buffer_state, batch_sequence
        )
        self.last_timestep.update(
            {
                "x": obs,
                "a": a,
            }
        )
        self.update()
        return a

    def end(self, reward: float, extra: Dict[str, Any]):
        # possibly process the reward
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        self.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(0),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), self.last_timestep
        )
        self.state.buffer_state = self.buffer.add(
            self.state.buffer_state, batch_sequence
        )
        self.update()
