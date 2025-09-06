from functools import partial
from typing import Any, Dict, Tuple
from dataclasses import replace

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from ml_instrumentation.Collector import Collector

import flashbax as fbx
import utils.chex as cxu
from algorithms.nn.NNAgent import AgentState as BaseAgentState
from algorithms.nn.NNAgent import Hypers as BaseHypers
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import huber_loss
from utils.policies import egreedy_probabilities

@cxu.dataclass
class Hypers(BaseHypers):
    target_refresh: int

@cxu.dataclass
class AgentState(BaseAgentState):
    target_params: Any
    hypers: Hypers
    carry: Any

def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        "delta": delta,
    }


class MADRQN(NNAgent):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        # set up the target network parameters
        self.target_refresh = params["target_refresh"]

        dummy_timestep = {
            "x": jnp.zeros(self.observations),
            "carry": jnp.zeros(self.hidden_size),
            "reset": jnp.bool(True),
            "last_a": jnp.int32(-1),
            "a": jnp.int32(0),
            "r": jnp.float32(0),
            "gamma": jnp.float32(0),
        }
        buffer_state = self.buffer.init(dummy_timestep)

        hypers = Hypers(
            **self.state.hypers.__dict__,
            target_refresh=params["target_refresh"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers" and k != "buffer_state"},
            target_params=self.state.params,
            buffer_state=buffer_state,
            carry=None,
            hypers=hypers,
        )

    def get_feature_function(self, builder: NetworkBuilder):
        return builder.getMultiplicativeActionRecurrentFeatureFunction()

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        self.q = builder.addHead(lambda: hk.Linear(self.actions, name="q"))

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array, last_a: jax.Array, carry: jax.Array = None):  # type: ignore
        phi = self.phi(state.params, x, a=last_a, carry=carry)
        return self.q(state.params, phi[0][:, -1]), phi[1][:, -1], phi[2]
    
    @partial(jax.jit, static_argnums=0)
    def _policy(self, state: AgentState, obs: jax.Array, last_a: jax.Array,) -> Tuple[jax.Array, jax.Array]:
        obs = jnp.expand_dims(obs, 0)
        q, carry, _ = self._values(state, obs, last_a, carry=state.carry)
        pi = egreedy_probabilities(q, self.actions, self.state.hypers.epsilon)[0]
        return pi, carry
    
    @partial(jax.jit, static_argnums=0)
    def act(
        self, state: AgentState, obs: jax.Array, last_a: jax.Array
    ) -> tuple[AgentState, jax.Array]:
        pi, state.carry = self._policy(state, obs, last_a)
        state.key, sample_key = jax.random.split(state.key)
        a = jax.random.choice(sample_key, self.actions, p=pi)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        state.updates += 1

        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, buffer_sample_key)

        state, metrics = self._computeUpdate(
            state, batch.experience, batch.probabilities
        )

        # Not doing Prioritized Buffer for now
        # priorities = metrics["delta"]
        # state.buffer_state = self.buffer.set_priorities(
        #     state.buffer_state, batch.indices, priorities
        # )

        state.target_params = jax.lax.cond(
            state.updates % state.hypers.target_refresh == 0,
            lambda: state.params,
            lambda: state.target_params,
        )

        return state

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch, weights)
        optimizer = optax.adam(**state.hypers.optimizer.__dict__)
        updates, optim = optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        return replace(state, params=params, optim=optim), metrics

    # Of shape: <batch, sequence, *feat>
    def _loss(
        self, params: hk.Params, target: hk.Params, batch: Dict, weights: jax.Array
    ):
        B, T = batch["a"][:, :-1].shape
        x = batch["x"][:, :-1]
        xp = batch["x"][:, 1:]
        a = batch["a"][:, :-1]
        last_a = batch["last_a"][:, :-1]
        r = batch["r"][:, :-1]
        g = batch["gamma"][:, :-1]
        carry = batch["carry"][:, :-1]
        carryp = batch["carry"][:, 1:]
        reset = batch["reset"][:, :-1]

        phi = self.phi(params, x, a=last_a, carry=carry, reset=reset, is_target=False)[0]
        phi_p = self.phi(target, xp, a=last_a, carry=carryp, reset=reset, is_target=True)[0]

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)
        
        qs = qs.reshape(-1, qs.shape[-1])
        qsp = qsp.reshape(-1, qsp.shape[-1])
        a = a.ravel()
        r = r.ravel()
        g = g.ravel()
        weights = jnp.repeat(weights, T)
        weights = weights.ravel()

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, metrics = batch_loss(qs, a, r, g, qsp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)

        return loss, metrics

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: jax.Array):
        state.carry = None
        state.last_timestep.update(
            {
                "last_a": jnp.int32(-1)
            }
        )
        state, a = self.act(state, obs, state.last_timestep["last_a"])
        state.last_timestep.update(
            {
                "x": obs,
                "a": a,
                "carry": jnp.zeros(self.hidden_size),   # Replaced with learnt init within alg
                "reset": jnp.bool(True)
            }
        )
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _step(self, state: AgentState, reward: jax.Array, obs: jax.Array, extra: Dict[str, jax.Array]):
        # see if the problem specified a discount term
        gamma = extra.get("gamma", 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(self.gamma * gamma),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        state.buffer_state = self.buffer.add(
            state.buffer_state, batch_sequence
        )
        state.last_timestep.update(
            {
                "last_a": state.last_timestep["a"]
            }
        )
        state, a = self.act(state, obs, jnp.array(state.last_timestep["last_a"]))
        state.last_timestep.update(
            {
                "x": obs,
                "a": a,
                "carry": state.carry[0],
                "reset": jnp.bool(False)
            }
        )
        state = self._maybe_update(state)
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
         # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(0)
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        state.buffer_state = self.buffer.add(
            state.buffer_state, batch_sequence
        )
        state = self._maybe_update(state)
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state
