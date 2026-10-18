"""Built-in node types (imported by :data:`pinneapple_flow.registry.REGISTRY` on first use)."""
from . import analysis, data, problem, solvers, surrogates, systems, twin_nodes, viz  # noqa: F401
from ..libcalls import library_specs
from ..registry import REGISTRY

for _spec in library_specs():
    REGISTRY.register(_spec)
