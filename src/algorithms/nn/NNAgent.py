from abc import abstractmethod
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent
from representations.networks import NetworkBuilder
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities
from algorithms.BaseAgent import AgentState as BaseAgentState, Hypers as BaseHypers


@cxu.dataclass
class OptimizerHypers:
    learning_rate: float
    b1: float
    b2: float
    eps: float


@cxu.dataclass
class Hypers(BaseHypers):
    epsilon: jax.Array
    optimizer: OptimizerHypers


@cxu.dataclass
class AgentState(BaseAgentState):
    params: Any
    optim: optax.OptState
    buffer_state: Any
    key: jax.Array
    last_timestep: Dict[str, jax.Array]
    steps: int
    updates: int
    hypers: Hypers


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

        self.epsilon_linear_decay = params.get("epsilon_linear_decay")
        self.total_steps = params["total_steps"]
        if self.epsilon_linear_decay is not None:
            self.initial_epsilon = params["initial_epsilon"]
            self.final_epsilon = params["final_epsilon"]
            epsilon = self.initial_epsilon
        else:
            epsilon = params["epsilon"]
        assert epsilon is not None or (
            self.epsilon_linear_decay is not None
            and self.initial_epsilon is not None
            and self.final_epsilon is not None
        )
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
        optimizer_hypers = OptimizerHypers(
            learning_rate=self.optimizer_params["alpha"],
            b1=self.optimizer_params["beta1"],
            b2=self.optimizer_params["beta2"],
            eps=self.optimizer_params["eps"],
        )
        optimizer = optax.adam(**optimizer_hypers.__dict__)
        opt_state = optimizer.init(net_params)

        # ------------------
        # -- Data ingress --
        # ------------------
        self.buffer_size = params["buffer_size"]
        self.batch_size = params["batch"]
        self.buffer_min_size = params.get("buffer_min_size", self.batch_size)
        if self.buffer_min_size == "buffer_size":
            self.buffer_min_size = self.buffer_size
        elif self.buffer_min_size == "batch_size":
            self.buffer_min_size = self.batch_size
        self.update_freq = params.get("update_freq", 1)
        self.priority_exponent = params.get("priority_exponent", 0.0)

        buffer = fbx.make_prioritised_trajectory_buffer(
            max_length_time_axis=self.buffer_size,
            min_length_time_axis=self.buffer_min_size,
            sample_batch_size=self.batch_size,
            add_batch_size=1,
            sample_sequence_length=self.n_step + 1,
            period=1,
            priority_exponent=self.priority_exponent,
        )
        self.buffer = replace(
            buffer,
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=(0,)),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
            set_priorities=jax.jit(buffer.set_priorities, donate_argnums=(0,)),
        )

        dummy_timestep = {
            "x": jnp.zeros(self.observations),
            "a": jnp.int32(0),
            "r": jnp.float32(0),
            "gamma": jnp.float32(0),
        }
        buffer_state = self.buffer.init(dummy_timestep)

        # --------------------------
        # -- Stateful information --
        # --------------------------
        hypers = Hypers(
            **self.state.hypers.__dict__,
            epsilon=epsilon,
            optimizer=optimizer_hypers,
        )
        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            params=net_params,
            optim=opt_state,
            buffer_state=buffer_state,
            key=self.key,
            last_timestep=dummy_timestep,
            steps=0,
            updates=0,
            hypers=hypers,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------

    @abstractmethod
    def _build_heads(self, builder: NetworkBuilder) -> None: ...

    @abstractmethod
    def _values(self, state: AgentState, x: jax.Array) -> jax.Array: ...

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _maybe_update(self, state: AgentState) -> AgentState: ...

    @partial(jax.jit, static_argnums=0)
    def _decay_epsilon(self, state: AgentState):
        epsilon = state.hypers.epsilon
        if self.epsilon_linear_decay is not None:
            decay_steps = self.epsilon_linear_decay * self.total_steps
            progress = state.steps / decay_steps
            calculated_epsilon = (
                self.initial_epsilon
                + (self.final_epsilon - self.initial_epsilon) * progress
            )
            epsilon = jnp.maximum(calculated_epsilon, self.final_epsilon)

        hypers = replace(state.hypers, epsilon=epsilon)
        return replace(state, hypers=hypers)

    def policy(self, obs: jax.Array) -> jax.Array:
        q = self.values(obs)
        pi = egreedy_probabilities(q, self.actions, self.state.hypers.epsilon)
        return pi

    @partial(jax.jit, static_argnums=0)
    def _policy(self, state: AgentState, obs: jax.Array) -> jax.Array:
        obs = jnp.expand_dims(obs, 0)
        q = self._values(state, obs)[0]
        pi = egreedy_probabilities(q, self.actions, state.hypers.epsilon)
        return pi

    @partial(jax.jit, static_argnums=0)
    def act(
        self,
        state: AgentState,
        obs: jax.Array,
    ) -> tuple[AgentState, jax.Array]:
        pi = self._policy(state, obs)
        state.key, sample_key = jax.random.split(state.key)
        a = jax.random.choice(sample_key, self.actions, p=pi)
        return state, a

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
        self.state, a = self._start(self.state, obs)
        return a

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: jax.Array):
        state, a = self.act(state, obs)
        state.last_timestep.update(
            {
                "x": obs,
                "a": a,
            }
        )
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state, a

    def step(self, reward: jax.Array, obs: jax.Array, extra: Dict[str, jax.Array]):
        self.state, a = self._step(self.state, reward, obs, extra)
        return a

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: jax.Array,
        extra: Dict[str, jax.Array],
    ):
        state, a = self.act(state, obs)

        # see if the problem specified a discount term
        gamma = extra.get("gamma", 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(state.hypers.gamma * gamma),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        buffer_state = self.buffer.add(state.buffer_state, batch_sequence)
        state = replace(state, buffer_state=buffer_state)
        state.last_timestep.update(
            {
                "x": obs,
                "a": a,
            }
        )
        state = self._maybe_update(state)
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state, a

    def end(self, reward: jax.Array, extra: Dict[str, jax.Array]):
        self.state = self._end(self.state, reward, extra)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
        # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(0),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        buffer_state = self.buffer.add(state.buffer_state, batch_sequence)
        state = replace(state, buffer_state=buffer_state)
        state = self._maybe_update(state)
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state
