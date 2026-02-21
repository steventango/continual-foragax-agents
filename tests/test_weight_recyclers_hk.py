import jax
import jax.numpy as jnp
import haiku as hk
import flax
import flax.linen as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.weight_recyclers_hk import (
    NeuronRecyclerScheduled as HKRecycler,
    get_flattened_dict,
)
from src.utils.weight_recyclers import NeuronRecyclerScheduled as FlaxRecycler
import optax
import numpy as np

# Create identical parameters and activations
key = jax.random.PRNGKey(42)


# Flax Setup
class FlaxNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        h1 = nn.Dense(features=16, name="layer1")(x)
        a1 = nn.relu(h1)
        # Store intermediate as Dopamine does:
        self.sow("intermediates", "layer1_act", a1)
        h2 = nn.Dense(features=4, name="layer2")(a1)
        return h2


flax_net = FlaxNet()
dummy_x = jnp.ones((2, 8))
variables = flax_net.init(key, dummy_x)
flax_params = variables["params"]
_, state = flax_net.apply(variables, dummy_x, mutable=["intermediates"])
flax_intermediates = state["intermediates"]
# The intermediates have __call__ key in Dopamine format if we manually specify
# Actually let's mock the intermediates to exactly what Dopamine expects
flax_intermediates_mock = {"layer1_act/__call__": (jax.nn.relu(jnp.ones((2, 16))),)}

flax_params_mock = flax.core.freeze(
    {
        "params": {
            "layer1": {
                "kernel": jax.random.normal(jax.random.PRNGKey(1), (8, 16)),
                "bias": jax.random.normal(jax.random.PRNGKey(2), (16,)),
            },
            "layer2": {
                "kernel": jax.random.normal(jax.random.PRNGKey(3), (16, 4)),
                "bias": jax.random.normal(jax.random.PRNGKey(4), (4,)),
            },
        }
    }
)
flax_opt = optax.adam(1e-3).init(flax_params_mock)

flax_recycler = FlaxRecycler(
    all_layers_names=["layer1", "layer2"],
    dead_neurons_threshold=-0.01,  # Negative threshold to trigger recycle
    reset_period=1,
    reset_start_step=0,
    reset_end_step=1000,
    score_type="redo",
    recycle_rate=0.5,
)

# Call flax recycle
flax_recycler._last_update_step = 10
new_flax_params, new_flax_opt = flax_recycler.update_weights(
    flax_intermediates_mock, flax_params_mock, key, flax_opt
)

# Haiku Setup
hk_params_mock = {
    "phi": {
        "layer1": {
            "w": flax_params_mock["params"]["layer1"]["kernel"],
            "b": flax_params_mock["params"]["layer1"]["bias"],
        },
        "layer2": {
            "w": flax_params_mock["params"]["layer2"]["kernel"],
            "b": flax_params_mock["params"]["layer2"]["bias"],
        },
    }
}
hk_intermediates_mock = {
    "layer1": jnp.ones((2, 16))  # pre relu activation
}
hk_opt = optax.adam(1e-3).init(hk_params_mock)

hk_recycler = HKRecycler(
    all_layers_names=["layer1", "layer2"],
    dead_neurons_threshold=-0.01,  # Negative threshold to trigger recycle
    reset_period=1,
    reset_start_step=0,
    reset_end_step=1000,
    score_type="redo",
    recycle_rate=0.5,
)

hk_recycler._last_update_step = 10
new_hk_params, new_hk_opt = hk_recycler.update_weights(
    hk_intermediates_mock, hk_params_mock, key, hk_opt
)

# Comparisons
new_flax_flat = flax.traverse_util.flatten_dict(new_flax_params, sep="/")
new_hk_flat = get_flattened_dict(new_hk_params, sep="/")

print("Flax params:")
for k, v in new_flax_flat.items():
    print(k, v.shape)

print("\nHaiku params:")
for k, v in new_hk_flat.items():
    print(k, v.shape)

# Check norms/differences
for l in ["layer1", "layer2"]:
    f_w = new_flax_flat[f"params/{l}/kernel"]
    h_w = new_hk_flat[f"phi/{l}/w"]
    diff_w = jnp.max(jnp.abs(f_w - h_w))
    print(f"{l} w diff: {diff_w}")
    assert diff_w < 1e-6

    f_b = new_flax_flat[f"params/{l}/bias"]
    h_b = new_hk_flat[f"phi/{l}/b"]
    diff_b = jnp.max(jnp.abs(f_b - h_b))
    print(f"{l} b diff: {diff_b}")
    assert diff_b < 1e-6

# Check Adam Opt state
flax_adam = new_flax_opt[0]
hk_adam = new_hk_opt[0]
assert type(flax_adam) == type(hk_adam)  # ScaleByAdamState

flax_mu_flat = flax.traverse_util.flatten_dict(flax_adam.mu, sep="/")
hk_mu_flat = get_flattened_dict(hk_adam.mu, sep="/")

for l in ["layer1", "layer2"]:
    diff_mu_w = jnp.max(
        jnp.abs(flax_mu_flat[f"params/{l}/kernel"] - hk_mu_flat[f"phi/{l}/w"])
    )
    diff_mu_b = jnp.max(
        jnp.abs(flax_mu_flat[f"params/{l}/bias"] - hk_mu_flat[f"phi/{l}/b"])
    )
    assert diff_mu_w < 1e-6
    assert diff_mu_b < 1e-6

# Check intersected dead neurons logs
flax_recycler.intersected_dead_neurons_with_last_reset(flax_intermediates_mock, 20000)
hk_recycler.intersected_dead_neurons_with_last_reset(hk_intermediates_mock, 20000)

flax_logs = flax_recycler.intersected_dead_neurons_with_last_reset(
    flax_intermediates_mock, 40000
)
hk_logs = hk_recycler.intersected_dead_neurons_with_last_reset(
    hk_intermediates_mock, 40000
)

print("Flax Logs:", flax_logs)
print("Haiku Logs:", hk_logs)

for k, v in flax_logs.items():
    if k.endswith("/__call__"):
        hk_key = k.replace("/__call__", "").split("/")[-1]
        hk_k = "/".join(k.split("/")[:-1]) + "/" + hk_key
        assert jnp.abs(v - hk_logs[hk_k]) < 1e-6

print("All tests passed!")
