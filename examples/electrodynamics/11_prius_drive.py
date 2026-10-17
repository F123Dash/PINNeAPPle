"""Drive model of the 2004 Toyota Prius motor from its FEA flux map: torque-speed envelope with MTPA and flux
weakening under inverter limits, efficiency map, and battery-side energy over the WLTC class 3b cycle.

Composes ``machines.IPMMachine`` (nonlinear FEA), ``machines.drive.FluxMap.from_fea`` (torque, dq flux and loss data
on an (i_d, i_q) grid), ``Drive`` and ``cycle_energy``. Assumptions, all stated in the output JSON: 250 A peak phase
current, 500 V DC link with space-vector PWM, phase resistance from the winding data at 100 degC (end-turn length
estimated), an illustrative compact car (not a Prius test vehicle). Mechanical, inverter and battery losses are
not modelled.

    python examples/electrodynamics/11_prius_drive.py           # about 20 min (flux map) the first time
    python examples/electrodynamics/11_prius_drive.py --quick   # coarse map
    python examples/electrodynamics/11_prius_drive.py --write-docs   # also the drive verification table of the guide
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
from pinneapple_simulation.electromagnetics.machines.drive import (
    Drive,
    FluxMap,
    Vehicle,
    cycle_energy,
    prius_phase_resistance,
    v_max_svpwm,
    wltc_class3b,
)

OUT = Path(__file__).parent / "results"
GUIDE = Path(__file__).parents[2] / "docs" / "guides" / "electromagnetics.md"
START, END = "<!-- drive-verification:start -->", "<!-- drive-verification:end -->"
RPM = 2 * math.pi / 60


def verification(quick: bool, write_docs: bool) -> list:
    from pinneapple_simulation.electromagnetics.machines import verification as mv
    from pinneapple_simulation.electromagnetics.verification import to_markdown

    names = [n for n in mv.CASES if n.startswith(("Drive", "Prius flux map"))]
    results = mv.run(names, quick=quick)
    table = to_markdown(results)
    notes = "\n".join(f"- {r.name}: {r.detail}" for r in results if r.detail)
    print(table + "\n" + notes)
    if write_docs:
        text = GUIDE.read_text()
        i, j = text.index(START) + len(START), text.index(END)
        GUIDE.write_text(text[:i] + "\n" + table + "\n\nNotes from the same run:\n\n" + notes + "\n" + text[j:])
    return [{"case": r.name, "error": r.error, "passed": r.passed, "detail": r.detail} for r in results]


def main(quick: bool) -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    cache = OUT / ("11_prius_fluxmap_quick.npz" if quick else "11_prius_fluxmap.npz")
    I_max, V_dc = 250.0, 500.0
    if cache.exists():
        fm = FluxMap.from_npz(cache)
    else:
        m = IPMMachine(prius_2004(), n_gap=288 if quick else 480)
        fm = FluxMap.from_fea(m, I_max, n_d=5 if quick else 9, n_q=5 if quick else 9, n_positions=6 if quick else 12,
                              loss_n=3 if quick else 5, loss_steps=12 if quick else 24,
                              progress=lambda s: print(s, flush=True))
        fm.to_npz(cache)
    R, wdata = prius_phase_resistance()
    drive = Drive(fm, R=R, I_max=I_max, V_max=v_max_svpwm(V_dc), n_d=121 if quick else 241)
    res: dict = {"assumptions": {"I_max_A": I_max, "V_dc_V": V_dc, "V_max_phase_peak_V": drive.V_max, **wdata,
                                 "flux_map": str(cache.name)}}

    speeds = np.linspace(0.0, 6000.0, 13 if quick else 25) * RPM
    env = [drive.max_torque(w) for w in speeds]
    res["envelope"] = {"rpm": (speeds / RPM).tolist(), "torque_Nm": [e["T"] for e in env],
                       "i_d": [e["i_d"] for e in env], "i_q": [e["i_q"] for e in env],
                       "power_kW": [e["T"] * w / 1e3 for e, w in zip(env, speeds, strict=True)]}
    print("envelope", [round(e["T"], 1) for e in env], flush=True)

    torques = np.linspace(-300.0, 340.0, 17 if quick else 33)
    em = drive.efficiency_map(speeds[1:], torques)
    eff = em["efficiency"]
    res["efficiency_map"] = {"rpm": (em["speeds"] / RPM).tolist(), "torque_Nm": torques.tolist(),
                             "efficiency": np.where(np.isfinite(eff), eff, -1.0).tolist(),
                             "peak_efficiency": float(np.nanmax(eff))}
    k = np.unravel_index(np.nanargmax(eff), eff.shape)
    res["efficiency_map"]["peak_at"] = {"rpm": float(em["speeds"][k[0]] / RPM), "torque_Nm": float(torques[k[1]])}

    t, v = wltc_class3b()
    veh = Vehicle(mass=1450.0, c_rr=0.009, cd_a=0.58, wheel_radius=0.30, gear_ratio=4.113, gear_efficiency=0.97)
    E = cycle_energy(drive, veh, t, v, envelope_speeds=13 if quick else 25)
    res["wltc_class3b"] = {k: E[k] for k in ("battery_J", "motoring_J", "regenerated_J", "drive_losses_J",
                                              "friction_brake_J", "wheel_positive_J", "distance_m",
                                              "battery_Wh_per_km")}
    res["vehicle"] = {"mass_kg": veh.mass, "c_rr": veh.c_rr, "cd_a_m2": veh.cd_a, "wheel_radius_m": veh.wheel_radius,
                      "gear_ratio": veh.gear_ratio, "gear_efficiency": veh.gear_efficiency,
                      "note": "illustrative compact car, not a measured Prius"}
    print(json.dumps(res["wltc_class3b"], indent=1), flush=True)
    _figures(res, em, E, speeds)
    res["seconds"] = time.perf_counter() - t0
    (OUT / "11_prius_drive.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def _figures(res, em, E, speeds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    rpm = em["speeds"] / RPM
    eff = em["efficiency"]
    S, T = np.meshgrid(rpm, em["torques"], indexing="ij")
    cs = ax[0].contourf(S, T, 100 * eff, levels=[50, 70, 80, 85, 88, 90, 92, 93, 94, 95, 96, 97, 98, 99],
                        cmap="viridis")
    ax[0].contour(S, T, 100 * eff, levels=[90, 94, 96], colors="w", linewidths=0.6)
    ax[0].plot(np.array(res["envelope"]["rpm"]), res["envelope"]["torque_Nm"], "k-", lw=1.5, label="envelope")
    ax[0].plot(rpm, -em["envelope_generating"], "k--", lw=1.0, label="braking envelope")
    ax[0].legend(loc="upper right")
    fig.colorbar(cs, ax=ax[0], label="efficiency (%)")
    ax[0].set_xlabel("speed (rpm)")
    ax[0].set_ylabel("torque (N m)")
    ax[0].set_title("Prius motor from FEA: 250 A, 500 V")
    ax[1].scatter(E["speed"] / RPM, E["torque"], s=3, c=E["t"], cmap="plasma")
    ax[1].plot(np.array(res["envelope"]["rpm"]), res["envelope"]["torque_Nm"], "k-", lw=1.0)
    ax[1].set_xlabel("speed (rpm)")
    ax[1].set_ylabel("torque (N m)")
    ax[1].set_title(f"WLTC class 3b operating points: {res['wltc_class3b']['battery_Wh_per_km']:.1f} Wh/km")
    fig.tight_layout()
    fig.savefig(OUT / "11_prius_drive.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    r = main("--quick" in sys.argv)
    r["verification"] = verification("--quick" in sys.argv, "--write-docs" in sys.argv)
    (OUT / "11_prius_drive.json").write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps({k: r[k] for k in ("assumptions", "wltc_class3b", "seconds")}, indent=1, default=float))
