from typing import Callable, NamedTuple, Optional
import haiku as hk
import chex
import jax
import jax.numpy as jnp
import jax.tree
import optax


def make_standalone_initializer(hk_initializer) -> Callable:
    """Convert a Haiku initializer to a standalone function.

    Haiku initializers expect to be called within a transform context.
    This wrapper creates a function that can be called with (key, shape, dtype).
    """

    def standalone_init(key, shape, dtype):
        def _inner():
            return hk_initializer(shape, dtype)

        transformed = hk.transform(_inner)
        return transformed.apply({}, key)

    return standalone_init


def compute_utility(
    p: chex.Array, grad: chex.Array, utility_name: str, key: chex.PRNGKey
) -> tuple[chex.Array, chex.PRNGKey]:
    """Compute utility of parameters for reinitialization."""
    if utility_name == "gradient":
        return jnp.abs(grad * p).flatten(), key
    elif utility_name == "magnitude":
        return jnp.abs(p).flatten(), key
    elif utility_name == "random":
        key, subkey = jax.random.split(key)
        return jax.random.uniform(subkey, shape=(p.size,)), key
    else:
        raise ValueError(f"Utility function not recognized: {utility_name}")


def prune_weights(
    utility: chex.Array, pruning_method: str, reinit_factor: float, key: chex.PRNGKey
) -> chex.Array:
    """Returns a boolean mask indicating which weights should be pruned.

    Returns a boolean array of the same size as utility, where True indicates
    a weight that should be reinitialized.
    """
    if pruning_method == "proportional":
        fraction_to_prune = utility.size * reinit_factor
        key, subkey = jax.random.split(key)
        drop_num = int(fraction_to_prune) + jax.random.bernoulli(
            subkey, fraction_to_prune % 1
        ).astype(jnp.int32)

        # Get sorted indices (lowest utility first)
        indices = jnp.argsort(utility)

        # Create mask: True for weights to reinitialize (lowest utility)
        mask = jnp.arange(utility.size) < drop_num

        # Map back to original parameter positions
        reinit_mask = jnp.zeros(utility.size, dtype=bool)
        reinit_mask = reinit_mask.at[indices].set(mask)

        return reinit_mask

    elif pruning_method == "threshold":
        prune_threshold = reinit_factor * jnp.mean(utility)
        return utility <= prune_threshold

    else:
        raise ValueError(f"Pruning method not recognized: {pruning_method}")


class SWRState(NamedTuple):
    """State for SelectiveWeightReinitialization optimizer."""

    step: chex.Array
    avg_utility: optax.Params
    rng_key: chex.PRNGKey
    reinit_indicator: chex.Array
    num_replaced: chex.Array


def selective_weight_reinitialization(
    utility_function: str,
    pruning_method: str,
    param_initializers: dict[str, Callable],
    reinit_freq: int = 0,
    reinit_factor: float = 0.0,
    decay_rate: float = 0.0,
    seed: int = 0,
) -> optax.GradientTransformation:
    """
    Optax GradientTransformation for Selective Weight Reinitialization.

    Arguments:
        utility_function: str in ["gradient", "magnitude", "random"]
        pruning_method: str in ["proportional", "threshold"]
        param_initializers: dict mapping parameter names to Haiku initializer callables
        reinit_freq: how often to reinitialize (in steps)
        reinit_factor: fraction/threshold for reinitialization
        decay_rate: exponential moving average decay for utility
        seed: random seed for reinitialization
    """

    # Convert Haiku initializers to standalone functions
    standalone_initializers = {
        name: make_standalone_initializer(init)
        for name, init in param_initializers.items()
    }

    def init_fn(params: optax.Params) -> SWRState:
        """Initialize optimizer state."""
        return SWRState(
            step=jnp.array(0, dtype=jnp.int32),
            avg_utility=jax.tree.map(lambda p: jnp.zeros(p.size), params),
            rng_key=jax.random.PRNGKey(seed),
            reinit_indicator=jnp.array(False),
            num_replaced=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: optax.Updates,
        state: SWRState,
        params: Optional[optax.Params] = None,
    ) -> tuple[optax.Updates, SWRState]:
        """Update parameters with selective reinitialization."""
        new_step = state.step + 1

        # Update moving average of utility
        if decay_rate > 0.0:
            new_avg_utility = jax.tree.map(
                # TODO: verify RNG handling
                lambda avg, p, grad: (
                    decay_rate * avg
                    + (1.0 - decay_rate)
                    * compute_utility(p, grad, utility_function, state.rng_key)[0]
                ),
                state.avg_utility,
                params,
                updates,
            )
        else:
            new_avg_utility = state.avg_utility

        # Check if it's time to reinitialize
        should_reinit = (reinit_freq > 0) and (new_step % reinit_freq == 0)

        def do_reinit(carry):
            """Perform reinitialization."""
            params, key, _ = carry

            def process_single_param(p, grad, avg_utility, initializer):
                if decay_rate > 0.0:
                    utility = avg_utility
                else:
                    # Note: We can't update key here in pure functional way inside tree_map
                    # This is a limitation - for now use the same key
                    utility, _ = compute_utility(p, grad, utility_function, key)

                _, subkey = jax.random.split(key)
                reinit_mask = prune_weights(
                    utility, pruning_method, reinit_factor, subkey
                )

                # Count number of weights to reinitialize
                num_reinit = jnp.sum(reinit_mask)

                # Generate new values for positions to reinitialize using the initializer
                _, subkey2 = jax.random.split(subkey)
                new_values = initializer(subkey2, p.shape, p.dtype)

                # Update parameters using the mask
                new_p = jnp.where(reinit_mask.reshape(p.shape), new_values, p)

                if decay_rate > 0.0:
                    # Reset utility for reinitialized weights
                    new_avg_u = jnp.where(reinit_mask, 0.0, avg_utility)
                else:
                    new_avg_u = avg_utility

                return new_p, new_avg_u, num_reinit

            results_tree = jax.tree.map(
                process_single_param,
                params,
                updates,
                new_avg_utility,
                standalone_initializers,
            )

            tuple_of_trees = jax.tree_util.tree_transpose(
                jax.tree_util.tree_structure(params),
                jax.tree_util.tree_structure((0, 0, 0)),
                results_tree,
            )
            new_params, new_avg_utility2, num_reinit_tree = tuple_of_trees

            total_replaced = jax.tree_util.tree_reduce(
                lambda x, y: x + y, num_reinit_tree, initializer=0
            )

            # Split key for next use
            new_key, _ = jax.random.split(key)
            return new_params, new_avg_utility2, new_key, total_replaced

        def no_reinit(carry):
            """Skip reinitialization."""
            params, key, num_replaced = carry
            return params, new_avg_utility, key, num_replaced

        new_params, new_avg_utility, new_key, num_replaced = jax.lax.cond(
            should_reinit,
            do_reinit,
            no_reinit,
            (params, state.rng_key, state.num_replaced),
        )

        # Create updates based on parameter changes
        new_updates = jax.tree.map(
            lambda new_p, old_p: new_p - old_p, new_params, params
        )

        new_state = SWRState(
            step=new_step,
            avg_utility=new_avg_utility,
            rng_key=new_key,
            reinit_indicator=jnp.array(should_reinit),
            num_replaced=num_replaced,
        )

        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore
