import jax
import jax.numpy as jnp
from utils.weight_recyclers_hk import NeuronRecyclerScheduled
import haiku as hk

recycler = NeuronRecyclerScheduled(
    all_layers_names=["phi"],
    reset_period=1000,
)

params = {"phi/w": jnp.ones((10, 5)), "phi_next/w": jnp.ones((5, 2))}
activations = {"phi": jnp.ones((64, 5))}
key = jax.random.PRNGKey(0)
opt_state = (0,) # dummy

@jax.jit
def update_call(step, p, acts, k, os):
    return recycler.maybe_update_weights(step, acts, p, k, os)

try:
    step = jnp.array(1000)
    print("Attempting to JIT recycler with tracer step...")
    update_call(step, params, activations, key, opt_state)
    print("Success!")
except Exception as e:
    print(f"JIT Failed: {e}")

