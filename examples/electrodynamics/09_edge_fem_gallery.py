"""Edge-element (Nedelec) FEM gallery: the modes of a circular waveguide, the resonances of a cylindrical cavity,
and induction heating of a billet in a coil. Each result is shown with its check against theory.

    python examples/electrodynamics/09_edge_fem_gallery.py           # about 20 s
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinneapple_simulation.electromagnetics import nedelec as ned  # noqa: E402
from pinneapple_simulation.electromagnetics import waves  # noqa: E402
from pinneapple_simulation.electromagnetics.analytic import cylinder_in_axial_field  # noqa: E402
from pinneapple_simulation.electromagnetics.constants import C0, MU0  # noqa: E402
from pinneapple_simulation.electromagnetics.mesh import Ring, polar_mesh  # noqa: E402
from pinneapple_simulation.numerical_solvers import eddy_current_fdm as fdm  # noqa: E402
from pinneapple_tools.visualization import electromagnetics as emviz  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
summary: dict = {}

# ------------------------------------------------------------------ 1. circular waveguide (radius 1 cm)
R = 0.01
n = 48
rings = [Ring(0.0, 1)] + [Ring(R * k / (n // 4), max(6, int(round(n * k / (n // 4))))) for k in range(1, n // 4 + 1)]
guide = polar_mesh(rings, lambda ly, r, p: "air")
space = ned.EdgeSpace.of(guide.points, guide.tris)
modes = ned.waveguide_modes(guide.points, guide.tris, 6)
exact = []
for f, kind, m, _ in waves.circular_waveguide_modes(R, 10):
    exact += [(f, kind)] * (2 if m > 0 else 1)
summary["circular_guide"] = [{"mode": kind, "fc_GHz": kc * C0 / (2 * math.pi) / 1e9, "exact_GHz": fe / 1e9,
                              "error": kc * C0 / (2 * math.pi) / fe - 1}
                             for (kc, kind), (fe, _) in zip(modes, sorted(exact)[:6], strict=True)]
k2, vecs = ned.eigenmodes(space, 4, return_vectors=True)
cent = guide.points[guide.tris].mean(1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, idx, name in ((axes[0], 0, "TE11"), (axes[1], 2, "TE21")):
    emviz.mesh_vector_field(guide.points, guide.tris, space.evaluate(vecs[:, idx], cent), ax=ax,
                            title=f"{name}: f_c = {math.sqrt(k2[idx]) * C0 / (2 * math.pi) / 1e9:.3f} GHz")
fig.tight_layout()
fig.savefig(OUT / "edge_fem_circular_guide_modes.png", dpi=130)

# ------------------------------------------------------------------ 2. cylindrical cavity (R = L = 1 cm)
tm = ned.extrude_to_tets(guide.points, guide.tris, np.linspace(0, R, n // 4 + 1))
f = np.sqrt(ned.eigenmodes(ned.EdgeSpace.of(tm.points, tm.tets), 3)) * C0 / (2 * math.pi)
ref = [waves.cylindrical_cavity_resonance("TM", 0, 1, 0, R, R), waves.cylindrical_cavity_resonance("TE", 1, 1, 1, R, R)]
summary["cylindrical_cavity"] = {"TM010_GHz": f[0] / 1e9, "TM010_exact_GHz": ref[0] / 1e9,
                                 "TE111_GHz": [f[1] / 1e9, f[2] / 1e9], "TE111_exact_GHz": ref[1] / 1e9}

# ------------------------------------------------------------------ 3. induction heating of a steel-free billet
a, sigma, H0 = 0.02, 1e6, 1e4                       # aluminium-alloy-like conductivity, 10 kA/m applied
r = np.linspace(0, 0.04, 161)
z = np.linspace(0, 0.04, 81)
RR, ZZ = np.meshgrid(r, z, indexing="ij")
sweep = []
for fkhz in (0.5, 1, 2, 5, 10, 20):
    w = 2 * math.pi * fkhz * 1e3
    Ar, P = cylinder_in_axial_field(r, a, sigma, w, H0)
    sig = np.where(RR < a - 1e-12, sigma, 0.0) + np.where(np.abs(RR - a) < 1e-12, 0.5 * sigma, 0.0)
    A = fdm.solve_axisymmetric_eddy_current(r, z, w, MU0 * np.ones_like(RR), sig, np.zeros_like(RR),
                                            np.repeat(Ar[:, None], len(z), axis=1))
    Pn = fdm.eddy_current_power(A, r, z, w, np.where(RR <= a + 1e-12, sigma, 0.0)) / z[-1]
    sweep.append({"f_kHz": fkhz, "a_over_delta": a / waves.skin_depth(w, sigma), "P_W_per_m": Pn,
                  "P_exact_W_per_m": P, "error": Pn / P - 1})
    if fkhz == 5:
        J = np.abs(fdm.eddy_current_density(A, w, np.where(RR <= a + 1e-12, sigma, 0.0)))
summary["induction_heating"] = sweep
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
pc = ax[0].pcolormesh(RR * 100, ZZ * 100, J ** 2 / (2 * sigma), shading="auto", cmap="inferno")
fig.colorbar(pc, ax=ax[0], label="Joule power density [W/m^3]")
ax[0].set_xlabel("r [cm]")
ax[0].set_ylabel("z [cm]")
ax[0].set_title("Billet at 5 kHz (a / delta = 2.8)")
ax[1].loglog([s["a_over_delta"] for s in sweep], [s["P_exact_W_per_m"] for s in sweep], "k-", label="Bessel solution")
ax[1].loglog([s["a_over_delta"] for s in sweep], [s["P_W_per_m"] for s in sweep], "o", label="FDM")
ax[1].set_xlabel("a / skin depth")
ax[1].set_ylabel("power per metre [W/m]")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT / "induction_heating_billet.png", dpi=130)

(OUT / "edge_fem_gallery.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
