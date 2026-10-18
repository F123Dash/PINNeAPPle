"""End effects of the Prius motor, outside the 2-D cross-section.

1. End-winding leakage inductance of both coil ends (Neumann's formula on the coil-end conductors,
   ``machines.endwinding``), in air and with the stator end face as an iron mirror; the end-turn length is the one
   of the phase resistance.
2. Its effect on the drive: torque envelope and efficiency at high speed with ``Drive(..., L_end=...)``.
3. End leakage of the magnet flux: a 3-D magnetostatic model of half the stack with air beyond the end face (gmsh +
   GetDP), against the same extruded model without it; the ratio corrects the 2-D magnet flux linkage, back-EMF and
   magnet torque.

    python examples/electrodynamics/16_prius_end_effects.py          # needs gmsh and getdp; about 30 min
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machine import belt_winding
from pinneapple_simulation.electromagnetics.machines import M400_50A, prius_2004
from pinneapple_simulation.electromagnetics.machines import getdp as G
from pinneapple_simulation.electromagnetics.machines.drive import (
    Drive,
    FluxMap,
    prius_phase_resistance,
    v_max_svpwm,
)
from pinneapple_simulation.electromagnetics.machines.endwinding import end_winding_inductance

OUT = Path(__file__).parent / "results"
RPM = 2 * math.pi / 60


def main() -> dict:
    t0 = time.perf_counter()
    g = prius_2004()
    ew = {ppm: end_winding_inductance(g, belt_winding(48, 4), 9.0, 6, 0.019366, pieces_per_metre=ppm)
          for ppm in (150, 300)}
    e = ew[300]
    print(f"end winding (both ends): L_aa {1e6 * e['air']['L_aa']:.1f} uH, M_ab {1e6 * e['air']['M_ab']:.1f} uH, "
          f"dq {1e6 * e['air']['L_end_dq']:.1f} uH in air; dq {1e6 * e['iron_face']['L_end_dq']:.1f} uH with the iron "
          f"face", flush=True)

    fm = FluxMap.from_npz(OUT / "11_prius_fluxmap.npz")
    R, _ = prius_phase_resistance(100.0)
    speeds = np.array([1000, 2000, 3000, 4000, 5000, 6000]) * RPM
    env = {}
    for label, L in (("2d", 0.0), ("air", e["air"]["L_end_dq"]), ("iron_face", e["iron_face"]["L_end_dq"])):
        d = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0), L_end=L)
        env[label] = [d.max_torque(w)["T"] for w in speeds]
        print(label, [round(x, 1) for x in env[label]], flush=True)
    d0 = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0))
    d1 = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0), L_end=e["air"]["L_end_dq"])
    eff = {}
    for rpm, T in ((4000, 100.0), (6000, 60.0)):
        a, b = d0.operating_point(T, rpm * RPM), d1.operating_point(T, rpm * RPM)
        eff[f"{rpm}rpm_{T:.0f}Nm"] = {"2d": a["efficiency"], "with_end_winding": b["efficiency"],
                                      "i_d_2d": a["i_d"], "i_d_end": b["i_d"]}

    leak = G.end_leakage_3d(g, M400_50A, 1.24, 1.05, n_gap=240, layers_stack=6, end_air=0.04, layers_air=6)
    k = leak["ratio"]
    print(f"3-D magnet flux under a pole with end air / extruded 2-D: {k:.4f}", flush=True)
    res = {"end_winding": {str(k_): v for k_, v in ew.items()}, "envelope_rpm": (speeds / RPM).tolist(),
           "envelope_Nm": env, "efficiency_points": eff, "magnet_end_leakage_3d": leak,
           "flux_correction": {"half_stack_m": g.length / 2, "ratio": k,
                               "note": "the 3-D half stack includes one end; the full stack has two, so the ratio "
                                       "applies to the whole machine"},
           "seconds": time.perf_counter() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / "16_prius_end_effects.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    main()
