import haiku as hk
import jax
import jax.numpy as jnp
from representations.networks import buildFeatureNetwork

key = jax.random.PRNGKey(0)
net, params, inits = buildFeatureNetwork((4, 4, 3), {'type': 'Forager2Net', 'hidden': 64}, key)
sample_in = jax.random.uniform(key, (32, 4, 4, 3))
out = net.apply(params, sample_in)

for layer_name, act in out.activations.items():
    if layer_name not in params: continue
    act = jax.nn.relu(act)
    reduce_axes = list(range(act.ndim - 1))
    score = jnp.mean(jnp.abs(act), axis=reduce_axes)
    score /= (jnp.mean(score) + 1e-9)
    print(f"{layer_name} score min: {jnp.min(score):.4f}, median: {jnp.median(score):.4f}, max: {jnp.max(score):.4f}")
    print(f"{layer_name} num dead (<0.01): {jnp.count_nonzero(score <= 0.01)}, "
          f"(<0.03): {jnp.count_nonzero(score <= 0.03)}, "
          f"(<0.05): {jnp.count_nonzero(score <= 0.05)}")
