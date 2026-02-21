import jax
import jax.numpy as jnp
from utils.weight_recyclers_hk import NeuronRecyclerScheduled
from representations.networks import buildFeatureNetwork
import haiku as hk

key = jax.random.PRNGKey(0)

net, feat_params, _ = buildFeatureNetwork((4, 4, 3), {'type': 'Forager2Net', 'hidden': 64}, key)
sample_in = jnp.zeros((32, 4, 4, 3))

params = {'phi': feat_params, 'q': {'q': {'w': jnp.ones((64, 4)), 'b': jnp.ones((4,))}}}

recycler = NeuronRecyclerScheduled(
    all_layers_names=["phi", "phi_2", "phi_3", "q"],
    reset_period=10,
    reset_start_step=0,
    reset_end_step=100_000,
    score_type="redo",
    recycle_rate=0.5,
)

out = net.apply(feat_params, sample_in)
intermediates = out.activations

import optax
optim = optax.chain(optax.scale_by_adam(), optax.scale(-0.01))
opt_state = optim.init(params)

new_params, new_opt_state = recycler.maybe_update_weights(
    10, intermediates, params, key, opt_state
)

def assert_mask_correctness():
    w_old = params['phi']['phi_2']['w']
    w_new = new_params['phi']['phi_2']['w']
    # phi_2 input is flat 256, output is 64. Weight shape is (256, 64)
    # The incoming mask zeroes columns in current_layer. So it zeroes elements along axis 1 (the 64 side).
    diff = jnp.abs(w_old - w_new) > 1e-5
    cols_changed = jnp.any(diff, axis=0) # Are entire columns zeroed?
    print(f"Incoming: changed {jnp.sum(cols_changed)} columns out of {cols_changed.shape[0]} in phi_2/w (shape {w_old.shape})")
    
    w_old_next = params['phi']['phi_3']['w']
    w_new_next = new_params['phi']['phi_3']['w']
    # phi_3 input is 64, output is 64. Weight shape is (64, 64)
    # The outgoing mask zeroes rows in next_layer. So it zeroes elements along axis 0 (the 64 side).
    diff_next = jnp.abs(w_old_next - w_new_next) > 1e-5
    rows_changed = jnp.any(diff_next, axis=1) # Are entire rows zeroed?
    print(f"Outgoing: changed {jnp.sum(rows_changed)} rows out of {rows_changed.shape[0]} in phi_3/w (shape {w_old_next.shape})")

assert_mask_correctness()
