"""Small JAX compatibility shim for third-party packages.

Some libraries (tensorflow_probability / distrax) access the removed
symbol ``jax.interpreters.xla.pytype_aval_mappings`` which existed
in older JAX versions. Newer JAX moved that mapping to
``jax.core.pytype_aval_mappings`` (removed in v0.7.0).

This module restores the old attribute as an alias if it's missing.
Importing this early (before importing distrax / tfp) prevents
the AttributeError raised during their import.
"""
from __future__ import annotations

try:
    import jax
    try:
        # jax.interpreters.xla may not exist in all JAX builds; guard access.
        import jax.interpreters.xla as _xla  # type: ignore
    except Exception:
        _xla = None

    if _xla is not None and not hasattr(_xla, "pytype_aval_mappings"):
        # Create the deprecated alias pointing to the new location.
        try:
            _xla.pytype_aval_mappings = jax.core.pytype_aval_mappings
        except Exception:
            # If anything goes wrong here, don't crash the program; leave it.
            pass
except Exception:
    # If jax isn't importable at all, nothing to do here.
    pass
