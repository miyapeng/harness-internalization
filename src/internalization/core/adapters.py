"""Resolve explicit project components without a retired ToyBackend fallback."""
from .interfaces import Components


def components_for(backend):
    if isinstance(backend, Components): return backend
    if hasattr(backend, "components"): return backend.components()
    raise TypeError("Backend must provide Components or components(); legacy backend fallback is retired")
