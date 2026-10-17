"""Temperatures of the Prius motor over repeated WLTC class 3b cycles (lumped network fed by the drive model's
losses), the time to reach the winding limit at peak load, and the demagnetisation margin of NdFeB grades at the
hottest magnet temperature with full negative d-axis current.

Needs the flux map of ``11_prius_drive.py`` (results/11_prius_fluxmap.npz). Losses per node: copper split between
slots and end windings by length, iron split between stator yoke, teeth and rotor by the FEA loss density at
the 250 A MTPA point, magnet loss to the magnets. Thermal data are the stated defaults of ``MachineThermalData``
(water jacket, coolant 65 degC): illustrative, not measured on a Prius.

    python examples/electrodynamics/13_prius_thermal.py            # about 10 min
"""
from __future__ import annotations

import json
import math
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
from pinneapple_simulation.electromagnetics.machines.thermal import (
    NDFEB_HCJ_MIN_KAM,
    demagnetisation_margin,
    intrinsic_coercivity,
    machine_network,
)

OUT = Path(__file__).parent / "results"


def iron_split(n_gap: int = 480) -> dict:
    """Fractions of the iron loss in the stator yoke, teeth and rotor at 250 A MTPA (49.5 deg), 1500 rpm."""
    m = IPMMachine(prius_2004(), n_gap=n_gap)
    gm = math.radians(49.5)
    r = m.losses(1500.0, -250 * math.sin(gm), 250 * math.cos(gm), steps=24)
    mesh = m.mm.mesh(0.0)
    area = mesh.areas()
    c = mesh.centroids()
    rc = np.hypot(c[:, 0], c[:, 1])
    st = m._stator_iron
    r_slot_bot = m.geometry.stator.r_in + m.geometry.stator.H0 + m.geometry.stator.H1 + m.geometry.stator.H2
    dens = r["stator_loss_density"]
    yoke = float(np.sum((area[st] * dens)[rc[st] >= r_slot_bot]))
    teeth = float(np.sum((area[st] * dens)[rc[st] < r_slot_bot]))
    rotor = float(np.sum(area[m._rotor_iron] * r["rotor_loss_density"]))
    tot = yoke + teeth + rotor
    return {"yoke": yoke / tot, "teeth": teeth / tot, "rotor": rotor / tot}


def main() -> dict:
    t0 = time.perf_counter()
    fm = FluxMap.from_npz(OUT / "11_prius_fluxmap.npz")
    R, wdata = prius_phase_resistance(100.0)
    drive = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0))
    t, v = wltc_class3b()
    veh = Vehicle(mass=1450.0, c_rr=0.009, cd_a=0.58, wheel_radius=0.30, gear_ratio=4.113, gear_efficiency=0.97)
    E = cycle_energy(drive, veh, t, v)
    split = iron_split()
    g = prius_2004()
    end = wdata["end_turn_per_side_m"]
    f_slot = g.length / (g.length + end)
    net = machine_network(g, end_length=end)

    def P_of(k):
        cu, fe, mg = E["loss_copper"][k], E["loss_iron"][k], E["loss_magnet"][k]
        return {"yoke": fe * split["yoke"], "teeth": fe * split["teeth"], "winding": cu * f_slot,
                "end_winding": cu * (1 - f_slot), "rotor": fe * split["rotor"], "magnets": mg}

    n = len(E["t"])
    P_cycle = np.array([[P_of(k)[nm] for nm in net.names] for k in range(n)])
    cycles = 4

    def P_t(time_s):
        return P_cycle[int(time_s) % n] if time_s < cycles * n else P_cycle[-1]

    T0 = net.boundaries[0][2]
    times, T = net.transient(P_t, cycles * n - 1, 1.0, T0)
    hot = {nm: float(T[:, i].max()) for i, nm in enumerate(net.names)}
    print("WLTC x4 peak temperatures (degC):", {k: round(v_, 1) for k, v_ in hot.items()}, flush=True)

    # peak load: 250 A MTPA at 1500 rpm, held from the end-of-cycle state until the winding reaches 180 degC
    op = drive.operating_point(float(drive.max_torque(1500 * 2 * math.pi / 60)["T"]) * 0.999, 1500 * 2 * math.pi / 60)
    Pk = {"yoke": op["iron"] * split["yoke"], "teeth": op["iron"] * split["teeth"], "winding": op["copper"] * f_slot,
          "end_winding": op["copper"] * (1 - f_slot), "rotor": op["iron"] * split["rotor"], "magnets": op["magnet"]}
    ts, Tp = net.transient([Pk[nm] for nm in net.names], 1800.0, 1.0, T[-1])
    iw = net.index("winding")
    reach = np.nonzero(Tp[:, iw] >= 180.0)[0]
    t_limit = float(ts[reach[0]]) if len(reach) else None
    T_steady_peak = dict(zip(net.names, map(float, net.steady([Pk[nm] for nm in net.names])), strict=True))
    print("peak load", round(op["T"], 1), "N m; time to 180 degC winding:", t_limit, "s", flush=True)

    # demagnetisation at the hottest magnet temperature of the run, full negative d current (250 A)
    T_mag = float(max(hot["magnets"], Tp[:, net.index("magnets")].max()))
    m = IPMMachine(prius_2004(), n_gap=480, magnet_temperature=T_mag)
    ms = m.solve(0.0, m.currents_dq(0.0, -250.0, 0.0))
    demag = {}
    for grade in ("N", "M", "H", "SH", "UH"):
        Hcj = intrinsic_coercivity(NDFEB_HCJ_MIN_KAM[grade] * 1e3, T_mag)
        demag[grade] = demagnetisation_margin(m, ms, Hcj)
        print(grade, f"Hcj {Hcj / 1e3:.0f} kA/m, worst reverse field {demag[grade]['H_rev_max'] / 1e3:.0f} kA/m, "
                     f"demagnetised fraction {demag[grade]['demagnetised_fraction']:.3f}", flush=True)
    res = {"iron_split": split, "wltc_cycles": cycles, "peak_temperatures_C": hot,
           "mean_losses_W": {"copper": float(np.mean(E["loss_copper"])), "iron": float(np.mean(E["loss_iron"])),
                             "magnet": float(np.mean(E["loss_magnet"]))},
           "peak_load": {"torque_Nm": op["T"], "rpm": 1500, "copper_W": op["copper"], "iron_W": op["iron"],
                         "magnet_W": op["magnet"], "time_to_180C_winding_s": t_limit,
                         "steady_state_C": T_steady_peak},
           "demagnetisation": {"magnet_temperature_C": T_mag, "i_d_A": -250.0, "grades": demag},
           "seconds": time.perf_counter() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / "13_prius_thermal.json").write_text(json.dumps(res, indent=1, default=float))
    _figure(times, T, net.names, ts, Tp, t_limit)
    return res


def _figure(times, T, names, ts, Tp, t_limit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, nm in enumerate(names):
        ax[0].plot(times / 60, T[:, i], label=nm)
        ax[1].plot(ts / 60, Tp[:, i], label=nm)
    ax[0].set_xlabel("time (min), 4 x WLTC class 3b")
    ax[0].set_ylabel("temperature (degC)")
    ax[0].legend(fontsize=8)
    ax[1].axhline(180, color="k", ls="--", lw=0.8)
    if t_limit is not None:
        ax[1].axvline(t_limit / 60, color="k", lw=0.8)
    ax[1].set_xlabel("time at peak load (min)")
    ax[1].set_title("250 A, 1500 rpm from the end-of-cycle state")
    fig.tight_layout()
    fig.savefig(OUT / "13_prius_thermal.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
