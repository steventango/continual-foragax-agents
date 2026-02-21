import jax
from representations.networks import buildFeatureNetwork
key = jax.random.PRNGKey(0)
net, params, _ = buildFeatureNetwork((4, 4, 3), {'type': 'Forager2Net', 'hidden': 64}, key)
def print_keys(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print_keys(v, prefix + k + "/")
        else:
            print(f"{prefix}{k}: {v.shape}")
print_keys(params)
