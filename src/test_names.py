import jax
import haiku as hk
import jax.numpy as jnp

def f():
    l1 = hk.Linear(10, name="phi")
    l2 = hk.Linear(10, name="phi")
    print(f"l1 name: {l1.name}")
    print(f"l2 name: {l2.name}")
    # Haiku assigns unique names internally, but f.name might be the same.
    # Let's check how Haiku tracks them.
    return l1(jnp.ones((1, 5))), l2(jnp.ones((1, 5)))

hk.transform(f).init(jax.random.PRNGKey(0))
