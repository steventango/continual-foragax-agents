"""NTK and churn metrics computation for PPO agents."""

import jax
import jax.numpy as jnp
import numpy as np
import logging
from typing import Tuple, Callable, NamedTuple, Any

logger = logging.getLogger(__name__)


def compute_ppo_value_metrics(
    agent_fn: Callable,
    params: Any,
    init_hstate: Any,
    x_ref: jnp.ndarray,
    action_dim: int,
    ntk_freq: int,
    update_idx: int,
    pred_before_churn: dict,
) -> Tuple[float, float, float]:
    """Compute NTK rank, condition number, and churn for PPO agents.

    Args:
        agent_fn: The network apply function (network.apply)
        params: Current network parameters (pytree)
        init_hstate: Initial hidden state for RNNs (None for feedforward)
        x_ref: Reference observations for metric computation, shape [n_samples, ...]
        action_dim: Number of actions (for encoding)
        ntk_freq: Frequency of metric computation (in updates)
        update_idx: Current update index (for determining if we should compute)
        pred_before_churn: Dict mapping to previous predictions for churn computation

    Returns:
        Tuple of (rank, condition_number, churn_norm) - use NaN for unavailable metrics
    """

    if ntk_freq <= 0 or update_idx % ntk_freq != 0:
        return np.nan, np.nan, np.nan

    try:
        # Cap at 100 samples for memory
        n_samples = min(100, len(x_ref))
        if n_samples < len(x_ref):
            sample_indices = np.linspace(0, len(x_ref) - 1, n_samples, dtype=int)
            x_ref_batch = x_ref[sample_indices]
        else:
            x_ref_batch = x_ref

        # Create dummy action encodings and reward traces (zeros)
        action_encoded_batch = jnp.zeros((n_samples, action_dim))
        reward_batch = jnp.zeros((n_samples, 1))
        sine_batch = jnp.zeros((n_samples, 1))
        cosine_batch = jnp.zeros((n_samples, 1))
        reward_trace_batch = jnp.zeros((n_samples, 1))

        # Create obs tuple in the format expected by PPO networks
        obs_tuple = (
            x_ref_batch,
            action_encoded_batch,
            reward_batch,
            sine_batch,
            cosine_batch,
            reward_trace_batch,
        )

        # Get value predictions
        _, pi, values = agent_fn(params, init_hstate, obs_tuple)
        pred_current = np.asarray(values)  # shape [n_samples]

        # Compute churn if we have previous predictions
        churn_norm = np.nan
        if 0 in pred_before_churn:
            pred_prev = pred_before_churn[0]
            churn_norm = float(np.linalg.norm(pred_current - pred_prev))
            logger.debug(f"Churn at update {update_idx}: {churn_norm:.6e}")

        # Store current predictions for next churn computation
        pred_before_churn[0] = pred_current

        # Compute NTK metrics using value function
        rank, cond = _compute_ntk_for_ppo(agent_fn, params, init_hstate, x_ref_batch,
                                          action_dim)

        logger.debug(f"PPO metrics at update {update_idx}: rank={rank}, cond={cond:.2e}, churn={churn_norm:.2e}")

        return float(rank), float(cond), churn_norm

    except Exception as e:
        logger.error(f"Failed to compute PPO metrics at update {update_idx}: {e}", exc_info=True)
        return np.nan, np.nan, np.nan


def _compute_ntk_for_ppo(
    agent_fn: Callable,
    params: Any,
    init_hstate: Any,
    x_ref_batch: jnp.ndarray,
    action_dim: int,
) -> Tuple[int, float]:
    """Compute NTK matrix rank and condition number for PPO value function.

    Args:
        agent_fn: The network apply function
        params: Current network parameters
        init_hstate: Initial hidden state
        x_ref_batch: Reference batch of observations, shape [n_samples, ...]
        action_dim: Number of actions

    Returns:
        Tuple of (rank, condition_number)
    """

    n_samples = len(x_ref_batch)

    # Create the value function as a function of parameters
    def values_fn(p, x):
        """Compute value for a single sample."""
        action_encoded = jnp.zeros((action_dim,))
        reward = jnp.zeros((1,))
        sine = jnp.zeros((1,))
        cosine = jnp.zeros((1,))
        reward_trace = jnp.zeros((1,))

        obs_tuple = (
            jnp.expand_dims(x, 0),  # Add batch dim
            jnp.expand_dims(action_encoded, 0),
            jnp.expand_dims(reward, 0),
            jnp.expand_dims(sine, 0),
            jnp.expand_dims(cosine, 0),
            jnp.expand_dims(reward_trace, 0),
        )

        _, _, value = agent_fn(p, init_hstate, obs_tuple)
        return value[0]  # Remove batch dim

    # Compute Jacobians over the batch
    jacobian_fn = jax.vmap(jax.jacrev(values_fn, argnums=0), in_axes=(None, 0))
    jacobians = jacobian_fn(params, x_ref_batch)

    # Flatten jacobians to [n_samples, n_params]
    def flatten_jacobians(jac_tree):
        leaves = jax.tree_util.tree_leaves(jac_tree)
        flat_leaves = [leaf.reshape(leaf.shape[0], -1) for leaf in leaves]
        return jnp.concatenate(flat_leaves, axis=1)

    jacobians_flat = flatten_jacobians(jacobians)

    # Compute NTK matrix
    ntk_matrix = jacobians_flat @ jacobians_flat.T

    # Compute rank and condition number
    rank = int(jnp.linalg.matrix_rank(ntk_matrix))
    cond_number = float(jnp.linalg.cond(ntk_matrix))

    return rank, cond_number


def collect_ppo_reference_observations(
    env,
    env_params,
    agent_fn: Callable,
    network_params: Any,
    init_hstate: Any,
    rng: jnp.ndarray,
    n_steps: int = 500,
    action_dim: int = 4,
) -> jnp.ndarray:
    """Collect reference observations from random environment rollouts.

    Args:
        env: Foragax environment
        env_params: Environment parameters
        agent_fn: Network apply function
        network_params: Initial network parameters
        init_hstate: Initial hidden state
        rng: Random number generator key
        n_steps: Number of environment steps to collect
        action_dim: Number of actions

    Returns:
        Array of observations, shape [n_steps, ...]
    """
    observations = []
    obs, env_state = env.reset(rng, env_params)

    for _ in range(n_steps):
        # Extract observation image
        if isinstance(obs, dict):
            obs_img = obs["image"]
        else:
            obs_img = obs

        observations.append(np.asarray(obs_img))

        # Random action
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (), 0, action_dim)

        # Step environment
        obs, env_state, reward, terminated, truncated, info = env.step(
            env_state, int(action), env_params
        )

        if terminated or truncated:
            obs, env_state = env.reset(jax.random.fold_in(rng, _), env_params)

    return jnp.stack(observations)
