from dataclasses import replace
from functools import partial
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers


@cxu.dataclass
class Hypers(BaseHypers):
    lambda_spectral: float


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


_SR_K = 2  # exponent k from the paper


def _power_iteration_sigma1(w_2d, num_iters=1):
    """Estimate σ₁(w_2d) via power iteration (1 step by default)."""
    u = jnp.ones((w_2d.shape[0],), dtype=w_2d.dtype)
    u = u / (jnp.linalg.norm(u) + 1e-12)

    def _step(u, _):
        v = w_2d.T @ u
        v = v / (jnp.linalg.norm(v) + 1e-12)
        u_new = w_2d @ v
        u_new = u_new / (jnp.linalg.norm(u_new) + 1e-12)
        return u_new, v

    u, vs = jax.lax.scan(_step, u, None, length=num_iters)
    v = vs[-1]
    sigma = u @ w_2d @ v
    return sigma


def _spectral_leaf(path, param):
    """Per-leaf spectral regularisation loss."""
    # Identify the leaf name from its path key
    leaf_name = (
        path[-1].key.lower() if hasattr(path[-1], "key") else str(path[-1]).lower()
    )

    # --- LayerNorm / normalisation scale parameters ---
    if leaf_name in ("scale", "gamma"):
        return jnp.sum(jnp.square(param - 1.0))

    # --- Bias / additive parameters ---
    if leaf_name in ("b", "bias", "beta", "offset") or param.ndim == 1:
        return jnp.linalg.norm(param) ** (2 * _SR_K)

    # --- Convolutional kernels (4-D) ---
    if param.ndim == 4:
        d_out = param.shape[-1]
        w_2d = jnp.transpose(param, (3, 0, 1, 2)).reshape((d_out, -1))
        sigma = _power_iteration_sigma1(w_2d)
        return jnp.square(sigma ** _SR_K - 1.0)

    # --- Dense / multiplicative weight matrices (2-D) ---
    if param.ndim == 2:
        sigma = _power_iteration_sigma1(param)
        return jnp.square(sigma ** _SR_K - 1.0)

    # Anything else (Scalars, etc.)
    return jnp.zeros((), dtype=param.dtype)


class DQN_Spectral_Reg(DQN):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        hypers = Hypers(
            **self.state.hypers.__dict__,
            lambda_spectral=params["lambda_spectral"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict):
        grad_fn = jax.grad(self._loss_spectral, has_aux=True)
        grad, metrics = grad_fn(
            state.params,
            state.target_params,
            batch,
            state.hypers.lambda_spectral,
        )
        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)
        updates, new_optim = optimizer.update(grad, state.optim, state.params)
            
        new_params = optax.apply_updates(state.params, updates)
        flat_updates, _ = ravel_pytree(updates)
        weight_change = jnp.linalg.norm(flat_updates, ord=1)
        metrics["weight_change"] = weight_change

        return replace(state, params=new_params, optim=new_optim), metrics

    def _loss_spectral(
        self,
        params: hk.Params,
        target: hk.Params,
        batch: Dict,
        lambda_spectral: float,
    ):
        q_loss, metrics = super()._loss(params, target, batch)

        # Apply spectral/L2 regularisation across the tree mapping
        spectral_losses = jax.tree_util.tree_map_with_path(
            _spectral_leaf, params
        )
        
        # Reduce to single scalar penalty
        reg_loss = lambda_spectral * jax.tree_util.tree_reduce(
            lambda a, b: a + b, spectral_losses
        )

        total_loss = q_loss + reg_loss
        return total_loss, metrics
