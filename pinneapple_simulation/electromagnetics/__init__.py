"""2D electromagnetics for electric machines: A_z magnetostatics on P1 triangles.

* :mod:`.mesh` -- polar ring mesher with a moving band for rotation; gmsh importer (optional dependency).
* :mod:`.materials` -- linear media, permanent magnets, nonlinear B-H curves.
* :mod:`.fem` -- assembly and Newton solver with Dirichlet and (anti-)periodic constraints.
* :mod:`.eddy` -- time-harmonic eddy currents in solid conductors with imposed net currents.
* :mod:`.postprocess` -- Maxwell-stress and co-energy torque, flux linkage, back-EMF, iron and magnet losses.
* :mod:`.machine` -- surface-PM / reluctance machine model (cogging, torque vs current angle, losses,
  rotor design sensitivities).
* :mod:`.analytic` -- closed-form references used by the tests.

See docs/core_concepts/motor_magnetics_and_topology_optimization.md.
"""
from .analytic import (
    Layer,
    RadialLayers,
    coax_field,
    magnetized_cylinder,
    skin_effect_wire,
    wire_field,
)
from .eddy import TimeHarmonicSolution, solve_time_harmonic
from .fem import Geometry, MagnetostaticSolution, constraint_map, flux_density, solve_magnetostatic
from .materials import (
    AIR,
    MU0,
    NU0,
    BHCurve,
    BrauerCurve,
    LinearMaterial,
    Magnet,
    illustrative_steel,
)
from .mesh import Mesh2D, Ring, mesh_from_gmsh, polar_mesh, zipper

__all__ = [
    "AIR", "MU0", "NU0", "BHCurve", "BrauerCurve", "Geometry", "Layer", "LinearMaterial", "Magnet",
    "MagnetostaticSolution", "Mesh2D", "RadialLayers", "Ring", "TimeHarmonicSolution", "coax_field",
    "constraint_map", "flux_density", "illustrative_steel", "magnetized_cylinder", "mesh_from_gmsh",
    "polar_mesh", "skin_effect_wire", "solve_magnetostatic", "solve_time_harmonic", "wire_field", "zipper",
]
