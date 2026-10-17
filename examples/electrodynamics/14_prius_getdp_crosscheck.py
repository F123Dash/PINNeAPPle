"""Cross-check of the Prius FEA against an independent code: gmsh + GetDP on the same geometry and materials.

For each quantity, both codes are evaluated at the same rotor angles and currents, and both are compared with the
published torque curve (Yang et al., IEEE IEMDC 2013):

* torque against current angle at 250 A (mean over 12 rotor angles in 60 electrical degrees);
* cogging over one cogging period;
* phase flux linkage over an electrical period and the back-EMF fundamental and THD at 1000 rpm;
* apparent L_d = (psi_d - psi_d0) / i_d and L_q = psi_q / i_q at the 250 A, 50 deg point (the same definition in
  both codes; ``psi_d0`` from the no-load solution).

Needs the ``gmsh`` and ``getdp`` executables (e.g. ``apt install gmsh getdp``).

    python examples/electrodynamics/14_prius_getdp_crosscheck.py          # about 1 h on one core
    python examples/electrodynamics/14_prius_getdp_crosscheck.py --quick  # coarse meshes, few points
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machine import belt_winding
from pinneapple_simulation.electromagnetics.machines import M400_50A, IPMMachine, prius_2004
from pinneapple_simulation.electromagnetics.machines import getdp as G
from pinneapple_simulation.electromagnetics.machines import verification as mv

OUT = Path(__file__).parent / "results"
TWO_PI = 2 * math.pi


def main(quick: bool) -> dict:
    t0 = time.perf_counter()
    g = prius_2004()
    n_gap = 480 if quick else 960
    m = IPMMachine(g, n_gap=n_gap)
    lay = belt_winding(48, 4)
    Br, mur = m.Br, m.magnet.mu_r

    def gd(theta, cur):
        return G.solve(g, theta, cur, M400_50A, Br, mur, lay, 9.0, n_gap=n_gap)

    npos = 6 if quick else 12
    phis = mv.YANG_2013_PHI0_DEG[::3] if quick else mv.YANG_2013_PHI0_DEG
    rows = []
    for p in phis:
        gm = math.radians(p - 90.0)
        T_o, T_g = [], []
        for th in m.positions(npos):
            cur = m.phase_currents(th, 250.0, gm)
            T_o.append(m.torque(m.solve(th, cur)))
            T_g.append(gd(th, cur)["torque"])
        ref = float(mv.YANG_2013_TORQUE_NM[list(mv.YANG_2013_PHI0_DEG).index(p)])
        rows.append({"phi0_deg": float(p), "pinneapple": float(np.mean(T_o)), "getdp": float(np.mean(T_g)),
                     "yang_2013": ref})
        r = rows[-1]
        print(f"Phi0 {p:5.0f}: PINNeAPPle {r['pinneapple']:7.2f}  GetDP {r['getdp']:7.2f}  published {ref:5.0f} N m",
              flush=True)

    th_c = m.positions(npos, period=m.cogging_period)
    cog_o = [m.torque(m.solve(t)) for t in th_c]
    cog_g = [gd(t, (0.0, 0.0, 0.0))["torque"] for t in th_c]

    nf = 24 if quick else 60
    th_e = m.positions(nf, period=TWO_PI / m.pole_pairs)
    psi_o = np.array([m.flux_linkage(m.solve(t)) for t in th_e])
    psi_g = np.array([gd(t, (0.0, 0.0, 0.0))["psi"] for t in th_e])
    emf_o = m.back_emf(psi_o, th_e, 1000.0)
    emf_g = m.back_emf(psi_g, th_e, 1000.0)

    def dq_mean(code, i_d, i_q):
        out = []
        for th in m.positions(npos):
            cur = m.currents_dq(th, i_d, i_q)
            psi = m.flux_linkage(m.solve(th, cur)) if code == "o" else gd(th, cur)["psi"]
            out.append(m.park(psi, th))
        return np.mean(out, axis=0)

    i_d, i_q = -250 * math.sin(math.radians(50)), 250 * math.cos(math.radians(50))
    L = {}
    for code in ("o", "g"):
        d0, _ = dq_mean(code, 0.0, 0.0)
        dd, _ = dq_mean(code, i_d, 0.0)
        _, qq = dq_mean(code, 0.0, i_q)
        L[code] = {"psi_d0_Wb": float(d0), "L_d_mH": 1e3 * float((dd - d0) / i_d), "L_q_mH": 1e3 * float(qq / i_q)}

    T_o = np.array([r["pinneapple"] for r in rows])
    T_g = np.array([r["getdp"] for r in rows])
    T_y = np.array([r["yang_2013"] for r in rows])
    res = {
        "n_gap": n_gap, "rotor_positions": npos,
        "torque_vs_angle": rows,
        "torque_difference_getdp_minus_pinneapple": {"max_abs_Nm": float(np.abs(T_g - T_o).max()),
                                                     "max_rel_at_peak": float(abs(T_g.max() - T_o.max()) / T_o.max()),
                                                     "rms_Nm": float(np.sqrt(np.mean((T_g - T_o) ** 2)))},
        "vs_yang": {"pinneapple_rms_Nm": float(np.sqrt(np.mean((T_o - T_y) ** 2))),
                    "getdp_rms_Nm": float(np.sqrt(np.mean((T_g - T_y) ** 2))),
                    "pinneapple_max_abs_Nm": float(np.abs(T_o - T_y).max()),
                    "getdp_max_abs_Nm": float(np.abs(T_g - T_y).max())},
        "cogging": {"pinneapple_pkpk_Nm": float(np.ptp(cog_o)), "getdp_pkpk_Nm": float(np.ptp(cog_g)),
                    "pinneapple": list(map(float, cog_o)), "getdp": list(map(float, cog_g))},
        "back_emf_1000rpm": {"pinneapple_fundamental_peak_V": emf_o["fundamental_peak"],
                             "getdp_fundamental_peak_V": emf_g["fundamental_peak"],
                             "pinneapple_thd": emf_o["thd"], "getdp_thd": emf_g["thd"], "positions": nf},
        "inductances_apparent_250A_50deg": {"pinneapple": L["o"], "getdp": L["g"]},
        "elements_getdp_example": gd(0.0, (0.0, 0.0, 0.0))["elements"], "elements_pinneapple": m.mm.n_elements,
        "seconds": time.perf_counter() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / "14_prius_getdp_crosscheck.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("torque_vs_angle", "cogging")}, indent=1, default=float))
    _figure(res, th_c, th_e, emf_o, emf_g, m)
    return res


def _figure(res, th_c, th_e, emf_o, emf_g, m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    rows = res["torque_vs_angle"]
    p = [r["phi0_deg"] for r in rows]
    ax[0].plot(p, [r["pinneapple"] for r in rows], "o-", label="PINNeAPPle")
    ax[0].plot(p, [r["getdp"] for r in rows], "x--", label="gmsh + GetDP")
    ax[0].plot(p, [r["yang_2013"] for r in rows], "s:", label="Yang et al. 2013")
    ax[0].set_xlabel("current angle Phi0 (deg)")
    ax[0].set_ylabel("mean torque at 250 A (N m)")
    ax[0].legend()
    ax[1].plot(np.degrees(th_c), res["cogging"]["pinneapple"], "o-", label="PINNeAPPle")
    ax[1].plot(np.degrees(th_c), res["cogging"]["getdp"], "x--", label="gmsh + GetDP")
    ax[1].set_xlabel("rotor angle (mech. deg)")
    ax[1].set_ylabel("cogging torque (N m)")
    ax[1].legend()
    el = np.degrees(th_e) * m.pole_pairs
    ax[2].plot(el, emf_o["e"][:, 0], "-", label="PINNeAPPle")
    ax[2].plot(el, emf_g["e"][:, 0], "--", label="gmsh + GetDP")
    ax[2].set_xlabel("electrical angle (deg)")
    ax[2].set_ylabel("phase A back-EMF at 1000 rpm (V)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "14_prius_getdp_crosscheck.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main("--quick" in sys.argv)
