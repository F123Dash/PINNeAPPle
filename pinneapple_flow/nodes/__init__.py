"""Built-in node types (imported by :data:`pinneapple_flow.registry.REGISTRY` on first use)."""
from ..libcalls import auto_specs, library_specs
from ..registry import REGISTRY
from . import analysis, data, problem, solvers, surrogates, systems, twin_nodes, viz  # noqa: F401

for _spec in library_specs() + auto_specs():
    REGISTRY.register(_spec)
