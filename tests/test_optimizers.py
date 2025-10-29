"""Tests for custom optimizers (SWR)."""

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pytest

from optimizers import (
    SWRState,
    compute_utility,
    prune_weights,
    selective_weight_reinitialization,
)


class TestUtilityFunctions:
    """Tests for utility computation functions."""

    def test_gradient_utility(self):
        """Test gradient-based utility computation."""
        p = jnp.array([1.0, -2.0, 3.0])
        grad = jnp.array([0.5, 1.0, -0.5])
        key = jax.random.PRNGKey(0)

        utility, new_key = compute_utility(p, grad, "gradient", key)

        expected = jnp.abs(p * grad)
        assert jnp.allclose(utility, expected.flatten())
        assert utility.shape == (3,)
        # Key should be returned unchanged for gradient utility
        assert jnp.array_equal(key, new_key)

    def test_magnitude_utility(self):
        """Test magnitude-based utility computation."""
        p = jnp.array([1.0, -2.0, 3.0])
        grad = jnp.zeros_like(p)  # Gradient shouldn't matter for magnitude
        key = jax.random.PRNGKey(0)

        utility, new_key = compute_utility(p, grad, "magnitude", key)

        expected = jnp.abs(p)
        assert jnp.allclose(utility, expected.flatten())
        assert utility.shape == (3,)
        assert jnp.array_equal(key, new_key)

    def test_random_utility(self):
        """Test random utility computation."""
        p = jnp.array([1.0, -2.0, 3.0])
        grad = jnp.zeros_like(p)
        key = jax.random.PRNGKey(42)

        utility, new_key = compute_utility(p, grad, "random", key)

        # Check utility is uniformly distributed
        assert utility.shape == (3,)
        assert jnp.all(utility >= 0.0) and jnp.all(utility <= 1.0)
        # Key should be updated for random utility
        assert not jnp.array_equal(key, new_key)

    def test_utility_flattens_output(self):
        """Test that utility functions flatten their output."""
        p = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        grad = jnp.ones_like(p)
        key = jax.random.PRNGKey(0)

        utility, _ = compute_utility(p, grad, "magnitude", key)

        assert utility.ndim == 1
        assert utility.shape == (4,)

    def test_invalid_utility_name(self):
        """Test that invalid utility names raise ValueError."""
        p = jnp.array([1.0, 2.0])
        grad = jnp.ones_like(p)
        key = jax.random.PRNGKey(0)

        with pytest.raises(ValueError, match="Utility function not recognized"):
            compute_utility(p, grad, "invalid", key)


class TestPruningFunctions:
    """Tests for weight pruning functions."""

    def test_proportional_pruning_basic(self):
        """Test proportional pruning with basic case."""
        utility = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        key = jax.random.PRNGKey(0)

        # Prune 40% (2 out of 5)
        mask = prune_weights(utility, "proportional", 0.4, key)

        # Count number of True values in mask
        num_pruned = jnp.sum(mask)

        # Should prune lowest utility values
        assert num_pruned == 2
        assert mask[0]  # utility 1.0
        assert mask[2]  # utility 2.0

    def test_proportional_pruning_zero_factor(self):
        """Test proportional pruning with zero factor."""
        utility = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        key = jax.random.PRNGKey(0)

        mask = prune_weights(utility, "proportional", 0.0, key)

        # Count number of True values in mask
        num_pruned = jnp.sum(mask)

        # Should prune nothing
        assert num_pruned == 0

    def test_proportional_pruning_full(self):
        """Test proportional pruning with factor >= 1.0."""
        utility = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        key = jax.random.PRNGKey(0)

        mask = prune_weights(utility, "proportional", 1.0, key)

        # Count number of True values in mask
        num_pruned = jnp.sum(mask)

        # Should prune all or nearly all
        assert num_pruned == 5

    def test_proportional_pruning_fractional(self):
        """Test proportional pruning with fractional number of weights."""
        utility = jnp.array([1.0, 5.0, 2.0])
        key = jax.random.PRNGKey(0)

        # 0.5 factor with 3 weights = 1.5 weights to prune
        # Should stochastically prune 1 or 2 weights
        mask = prune_weights(utility, "proportional", 0.5, key)

        # Count number of True values in mask
        num_pruned = jnp.sum(mask)

        assert num_pruned in [1, 2]

    def test_threshold_pruning_basic(self):
        """Test threshold-based pruning."""
        utility = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        key = jax.random.PRNGKey(0)

        # Mean is 3.0, threshold is 0.5 * 3.0 = 1.5
        mask = prune_weights(utility, "threshold", 0.5, key)

        # Should prune values < 1.5 (only index 0 with utility 1.0)
        expected_mask = jnp.array([True, False, False, False, False])
        assert jnp.array_equal(mask, expected_mask)

    def test_threshold_pruning_high_threshold(self):
        """Test threshold pruning with high threshold."""
        utility = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        key = jax.random.PRNGKey(0)

        # Mean is 3.0, threshold is 2.0 * 3.0 = 6.0
        mask = prune_weights(utility, "threshold", 2.0, key)

        # Count number of True values in mask
        num_pruned = jnp.sum(mask)

        # Should prune all values
        assert num_pruned == 5

    def test_threshold_pruning_low_threshold(self):
        """Test threshold pruning with very low threshold."""
        utility = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        key = jax.random.PRNGKey(0)

        # Mean is 3.0, threshold is 0.1 * 3.0 = 0.3
        mask = prune_weights(utility, "threshold", 0.1, key)

        # Count number of True values in mask
        num_pruned = jnp.sum(mask)

        # Should prune no values (all > 0.3)
        assert num_pruned == 0

    def test_invalid_pruning_method(self):
        """Test that invalid pruning methods raise ValueError."""
        utility = jnp.array([1.0, 2.0, 3.0])
        key = jax.random.PRNGKey(0)

        with pytest.raises(ValueError, match="Pruning method not recognized"):
            prune_weights(utility, "invalid", 0.5, key)


class TestSWROptimizer:
    """Tests for the complete SWR optimizer."""

    @pytest.fixture
    def simple_params(self):
        """Simple parameter tree for testing."""
        return {"w": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])}

    @pytest.fixture
    def simple_config(self):
        """Simple SWR configuration."""
        return {
            "utility_function": "magnitude",
            "pruning_method": "proportional",
            "param_initializers": {"w": hk.initializers.TruncatedNormal(stddev=1.0)},
            "reinit_freq": 10,
            "reinit_factor": 0.2,
            "decay_rate": 0.0,
            "seed": 42,
        }

    def test_swr_initialization(self, simple_params, simple_config):
        """Test SWR optimizer initialization."""
        swr = selective_weight_reinitialization(**simple_config)
        state = swr.init(simple_params)

        assert isinstance(state, SWRState)
        assert state.step == 0
        assert not state.reinit_indicator
        assert state.num_replaced == 0
        # Check avg_utility has correct shape
        assert jax.tree.structure(state.avg_utility) == jax.tree.structure(
            simple_params
        )

    def test_swr_no_reinit_before_frequency(self, simple_params, simple_config):
        """Test that SWR doesn't reinitialize before reaching frequency."""
        swr = selective_weight_reinitialization(**simple_config)
        state = swr.init(simple_params)

        # Apply updates for fewer steps than reinit_freq
        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}
        for _ in range(5):
            updates, state = swr.update(updates, state, simple_params)

        # Parameters should not have been reinitialized
        assert isinstance(state, SWRState)
        assert state.step == 5
        assert not state.reinit_indicator

    def test_swr_reinit_at_frequency(self, simple_params, simple_config):
        """Test that SWR reinitializes at the specified frequency."""
        swr = selective_weight_reinitialization(**simple_config)
        state = swr.init(simple_params)

        # Apply updates until reinit_freq
        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}
        for _ in range(10):
            updates, state = swr.update(updates, state, simple_params)

        # Should have reinitialized at step 10
        assert isinstance(state, SWRState)
        assert state.step == 10
        assert state.reinit_indicator
        assert state.num_replaced > 0  # Some weights should be replaced

    def test_swr_zero_frequency_no_reinit(self, simple_params, simple_config):
        """Test that zero frequency prevents reinitialization."""
        config = {**simple_config, "reinit_freq": 0}
        swr = selective_weight_reinitialization(**config)
        state = swr.init(simple_params)

        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}
        for _ in range(20):
            updates, state = swr.update(updates, state, simple_params)

        # Should never reinitialize
        assert isinstance(state, SWRState)
        assert not state.reinit_indicator
        assert state.num_replaced == 0

    def test_swr_with_decay_rate(self, simple_params, simple_config):
        """Test SWR with exponential moving average of utility."""
        config = {**simple_config, "decay_rate": 0.9}
        swr = selective_weight_reinitialization(**config)
        state = swr.init(simple_params)

        updates = {"w": jnp.array([-0.1, -0.2, -0.3, -0.4, -0.5])}
        for _ in range(5):
            updates, state = swr.update(updates, state, simple_params)

        # avg_utility should be non-zero after updates
        assert isinstance(state, SWRState)
        assert jnp.any(state.avg_utility["w"] != 0.0)

    def test_swr_gradient_utility(self, simple_params, simple_config):
        """Test SWR with gradient utility function."""
        config = {**simple_config, "utility_function": "gradient"}
        swr = selective_weight_reinitialization(**config)
        state = swr.init(simple_params)

        # Provide gradients (updates)
        updates = {"w": jnp.array([-1.0, -0.5, -0.1, -0.2, -0.3])}
        for _ in range(10):
            updates, state = swr.update(updates, state, simple_params)

        # Should reinitialize based on gradient * param magnitude
        assert isinstance(state, SWRState)
        assert state.reinit_indicator

    def test_swr_threshold_pruning(self, simple_params, simple_config):
        """Test SWR with threshold-based pruning."""
        config = {**simple_config, "pruning_method": "threshold", "reinit_factor": 0.5}
        swr = selective_weight_reinitialization(**config)
        state = swr.init(simple_params)

        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}
        for _ in range(10):
            updates, state = swr.update(updates, state, simple_params)

        assert isinstance(state, SWRState)
        assert state.reinit_indicator

    def test_swr_uniform_reinit(self, simple_params, simple_config):
        """Test SWR with uniform reinitialization."""
        config = {
            **simple_config,
            "param_initializers": {"w": hk.initializers.UniformScaling(scale=0.5)},
        }
        swr = selective_weight_reinitialization(**config)
        state = swr.init(simple_params)

        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}
        for _ in range(10):
            new_updates, state = swr.update(updates, state, simple_params)

        # Should have reinitialized with uniform distribution
        assert isinstance(state, SWRState)
        assert state.reinit_indicator
        assert state.num_replaced > 0

    def test_swr_chain_with_adam(self, simple_params, simple_config):
        """Test SWR chained with ADAM optimizer."""
        swr = selective_weight_reinitialization(**simple_config)
        adam = optax.adam(learning_rate=0.001)
        optimizer = optax.chain(swr, adam)

        state = optimizer.init(simple_params)
        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}

        # Apply several updates
        params = simple_params
        for _ in range(15):
            updates_out, state = optimizer.update(updates, state, params)
            params = optax.apply_updates(params, updates_out)

        # Should work without errors

    def test_swr_multiple_param_groups(self):
        """Test SWR with multiple parameter groups."""
        params = {
            "layer1": jnp.array([1.0, 2.0, 3.0]),
            "layer2": jnp.array([4.0, 5.0, 6.0, 7.0]),
        }
        config = {
            "utility_function": "magnitude",
            "pruning_method": "proportional",
            "param_initializers": {
                "layer1": hk.initializers.TruncatedNormal(stddev=1.0),
                "layer2": hk.initializers.UniformScaling(scale=2.0),
            },
            "reinit_freq": 5,
            "reinit_factor": 0.3,
            "decay_rate": 0.0,
            "seed": 42,
        }
        swr = selective_weight_reinitialization(**config)
        state = swr.init(params)

        updates = {
            "layer1": jnp.array([-0.1, -0.2, -0.3]),
            "layer2": jnp.array([-0.1, -0.2, -0.3, -0.4]),
        }

        for _ in range(5):
            updates_out, state = swr.update(updates, state, params)

        # Should reinitialize both layers
        assert isinstance(state, SWRState)
        assert state.reinit_indicator

    def test_swr_reproducibility(self, simple_params, simple_config):
        """Test that SWR produces reproducible results with same seed."""
        swr1 = selective_weight_reinitialization(**simple_config)
        swr2 = selective_weight_reinitialization(**simple_config)

        state1 = swr1.init(simple_params)
        state2 = swr2.init(simple_params)

        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}
        params = simple_params

        # Run both for same number of steps
        for _ in range(10):
            updates1, state1 = swr1.update(updates, state1, params)
            updates2, state2 = swr2.update(updates, state2, params)

        # Should produce identical results
        assert jnp.allclose(updates1["w"], updates2["w"])
        assert isinstance(state1, SWRState)
        assert isinstance(state2, SWRState)
        assert state1.num_replaced == state2.num_replaced

    def test_swr_different_seeds(self, simple_params, simple_config):
        """Test that different seeds produce different reinitializations."""
        config1 = {**simple_config, "seed": 42}
        config2 = {**simple_config, "seed": 123}

        swr1 = selective_weight_reinitialization(**config1)
        swr2 = selective_weight_reinitialization(**config2)

        state1 = swr1.init(simple_params)
        state2 = swr2.init(simple_params)

        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}
        params = simple_params

        for _ in range(10):
            updates1, state1 = swr1.update(updates, state1, params)
            updates2, state2 = swr2.update(updates, state2, params)

        # Different seeds may produce different reinitializations
        # (not guaranteed to be different, but likely with random utility)
        assert not jnp.allclose(updates1["w"], updates2["w"])

    def test_swr_jit_compatibility(self, simple_params, simple_config):
        """Test that SWR works with JAX JIT compilation."""
        swr = selective_weight_reinitialization(**simple_config)

        @jax.jit
        def update_step(updates, state, params):
            return swr.update(updates, state, params)

        state = swr.init(simple_params)
        updates = {"w": jnp.array([-0.1, -0.1, -0.1, -0.1, -0.1])}

        # Should work with JIT
        for _ in range(10):
            updates, state = update_step(updates, state, simple_params)

        assert isinstance(state, SWRState)
        assert state.step == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
