import haiku as hk
import jax
import jax.numpy as jnp
from representations.networks import buildFeatureNetwork

key = jax.random.PRNGKey(0)
net, params, inits = buildFeatureNetwork((4, 4, 3), {'type': 'Forager2Net', 'hidden': 64}, key)
sample_in = jax.random.uniform(key, (32, 4, 4, 3))
out = net.apply(params, sample_in)

total_neurons = 0.0
total_dead = 0.0
for layer_name, act in out.activations.items():
    if layer_name not in params["phi"]:
        continue
    act = jax.nn.relu(act)
    reduce_axes = list(range(act.ndim - 1))
    score = jnp.mean(jnp.abs(act), axis=reduce_axes)
    score /= jnp.mean(score) + 1e-9
    
    dead_count = jnp.count_nonzero(score <= 0.01)
    layer_size = float(jnp.size(score))
    
    print(f"Layer {layer_name}: {dead_count} dead / {layer_size} total ({(dead_count/layer_size)*100:.2f}%)")
    
    total_neurons += layer_size
    total_dead += dead_count

print("Total dead percentage:", (total_dead / total_neurons) * 100.0 if total_neurons > 0 else 0)
