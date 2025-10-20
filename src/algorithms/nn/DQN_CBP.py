from dataclasses import replace
from functools import partial
from typing import Any, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.NNAgent import AgentState as BaseAgentState
from algorithms.nn.NNAgent import Hypers as BaseHypers
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import huber_loss


def init_cbp_state(params: jax.Array):
    feats = params['phi']
    ages = {name: jnp.zeros(feats[name]['w'].shape[1]) for name in feats}
    utils = {name: jnp.zeros(feats[name]['w'].shape[1]) for name in feats}
    counts = {name: jnp.zeros((1,)) for name in feats}
    return ages, utils, counts
@cxu.dataclass
class Hypers(BaseHypers):
    target_refresh: int
    replacement_rate: float
    decay_rate: float
    maturity_threshold: int


@cxu.dataclass
class AgentState(BaseAgentState):
    target_params: Any
    init_params: Any
    hypers: Hypers
    utils: Any
    ages: Any
    counts: Any


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        "delta": delta,
    }


class DQN_CBP(NNAgent):
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
        hypers = Hypers(
            **self.state.hypers.__dict__,
            target_refresh=params["target_refresh"],
            replacement_rate=params["replacement_rate"],
            decay_rate=params["decay_rate"],
            maturity_threshold=params["maturity_threshold"],
        )

        ages, utils, counts = init_cbp_state(self.state.params)
        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            target_params=self.state.params,
            init_params=self.state.params,
            hypers=hypers,
            utils=utils,
            ages=ages,
            counts=counts,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        def q_net_builder():
            return hk.Linear(self.actions, name="q")

        self.q_net, _, self.q = builder.addHead(q_net_builder, name="q")

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):  # type: ignore
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        updates = state.updates + 1

        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, buffer_sample_key)

        state, metrics, activations = self._computeUpdate(state, batch.experience)
        state = self._selectiveReinitialize(state, activations)
        target_params = self._update_target_network(state, updates)

        return replace(
            state,
            updates=updates,
            target_params=target_params,
        ), metrics

    @partial(jax.jit, static_argnums=0)
    def _update_target_network(self, state: AgentState, updates: int):
        target_params = jax.lax.cond(
            updates % state.hypers.target_refresh == 0,
            lambda: state.params,
            lambda: state.target_params,
        )
        return target_params

    # --------------------------------
    # -- Selective Reinitialization --
    # --------------------------------
    def _selectiveReinitialize(self, state: AgentState, activations):
        # pop flatten layer activations
        # TODO: Hard coded for ForagerNetLinear
        activations.pop('phi')

        # average mini-batch activations
        activations = jax.tree.map(partial(jnp.mean, axis=0), activations)

        # update unit age
        new_ages = jax.tree.map(lambda x: x + 1, state.ages)

        # compute unit utility
        # TODO: Hard coded for ForagerNetLinear
        def get_output_weight_mags(path, x):
            if path[0].key == 'phi_1':
                return jnp.abs(state.params['phi']['phi_2']['w']).sum(axis=-1)
            elif path[0].key == 'phi_2':
                return jnp.abs(state.params['q']['q']['w']).sum(axis=-1)
            else:
                return jnp.zeros_like(x)


        output_weight_mags = jax.tree_util.tree_map_with_path(get_output_weight_mags, state.utils)
        u = jax.tree.map(lambda h, w: jnp.abs(h) + w, activations, output_weight_mags)
        new_utils = jax.tree.map(lambda ut, u: state.hypers.decay_rate * ut + (1 - state.hypers.decay_rate) * u, state.utils, u)


        # update number of units to replace
        n_eligible = jax.tree.map(lambda x: (x >= state.hypers.maturity_threshold).sum(), new_ages)
        new_counts = jax.tree.map(lambda c, n_eligible: c + n_eligible * state.hypers.replacement_rate, state.counts, n_eligible)

        def identify_reinit_idx(util, count):
            return jax.lax.cond(
                (count < 1)[0],
                lambda: -1,
                lambda: jnp.argmin(util),
            )


        reinit_idx = jax.tree.map(identify_reinit_idx, new_utils, new_counts)

        # reinit its input weights (back to initial weights)
        def reinit_input_params(params, init_params, idx):
            def _reinit_input_params(params, init_params, idx):
                w = params['w']
                init_w = init_params['w']
                new_w = w.at[:, idx].set(init_w[:, idx])

                b = params['b']
                init_b = init_params['b']
                new_b = b.at[idx].set(init_b[idx])

                return {'w': new_w, 'b': new_b}

            return jax.lax.cond(
                idx == -1,
                lambda: params,
                lambda: _reinit_input_params(params, init_params, idx),
            )

        new_params = {
            'phi': {
                name: reinit_input_params(
                    state.params['phi'][name],
                    state.init_params['phi'][name],
                    reinit_idx[name],
                ) for name in state.params['phi'].keys()},
            'q': state.params['q'],
        }

        # set output weights to zero
        ...


        # set utility and age to zero
        def reset_to_zero(x, idx):
            return jax.lax.cond(
                idx == -1,
                lambda: x,
                lambda: x.at[idx].set(0.0),
            )

        new_utils = jax.tree.map(reset_to_zero, new_utils, reinit_idx)
        new_ages = jax.tree.map(reset_to_zero, new_ages, reinit_idx)

        # set optimizer state to zero
        ...


        # decrement number of units to replace
        def decrement_counts(count, idx):
            return jax.lax.cond(
                idx == -1,
                lambda: count,
                lambda: count - 1,
            )

        new_counts = jax.tree.map(decrement_counts, new_counts, reinit_idx)

        return replace(
            state,
            params=new_params,
            utils=new_utils,
            ages=new_ages,
            counts=new_counts,
        )

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, (metrics, activations) = grad_fn(state.params, state.target_params, batch)
        optimizer = optax.adam(**state.hypers.optimizer.__dict__)

        new_params = {}
        new_optim = {}
        weight_change = 0
        for name, p in state.params.items():
            updates, optim = optimizer.update(grad[name], state.optim[name], p)
            new_params[name] = optax.apply_updates(p, updates)
            new_optim[name] = optim
            flat_updates, _ = ravel_pytree(updates)
            weight_change += jnp.linalg.norm(flat_updates, ord=1)
        metrics["weight_change"] = weight_change

        return replace(state, params=new_params, optim=new_optim), metrics, activations

    def _loss(self, params: hk.Params, target: hk.Params, batch: Dict):
        x = batch["x"][:, 0]
        xp = batch["x"][:, -1]
        a = batch["a"][:, 0]
        rs = batch["r"]
        gs = batch["gamma"]
        gs = jnp.concatenate([jnp.ones((gs.shape[0], 1)), gs[:, :-1]], axis=1)
        gs = jnp.cumprod(gs, axis=1)

        r = jnp.sum(rs[:, :-1] * gs[:, :-1], axis=1)
        g = gs[:, -1]

        _forward = self.phi(params, x)
        phi = _forward.out
        phi_p = self.phi(target, xp).out

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, batch_metrics = batch_loss(qs, a, r, g, qsp)

        loss = jnp.mean(losses)

        # aggregate metrics
        metrics = {
            "loss": loss,
            "abs_td_error": jnp.mean(jnp.abs(batch_metrics["delta"])),
            "squared_td_error": jnp.mean(jnp.square(batch_metrics["delta"])),
        }

        return loss, (metrics, _forward.activations)
