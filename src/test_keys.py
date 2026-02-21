import haiku as hk
import jax
import jax.numpy as jnp
from representations.networks import buildFeatureNetwork

key = jax.random.PRNGKey(0)
net, params, inits = buildFeatureNetwork((4, 4, 3), {'type': 'Forager2Net', 'hidden': 64}, key)
print("Params keys:", params.keys())

out = net.apply(params, jnp.zeros((1, 4, 4, 3)))
print("Activations keys:", out.activations.keys())
