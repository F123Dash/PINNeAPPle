"""V-magnet rotor of the Prius motor: FEA + Gaussian-process surrogate + NSGA-II, against NSGA-II on the FEA alone.

Five rotor variables (magnet width and thickness, V depth and opening, bridge), two objectives (maximise torque at
250 A and 50 degrees, minimise magnet mass), torque ripple at most 30 %. The hybrid study (Latin hypercube start,
rounds of GP fit -> NSGA-II on the GP -> FEA on points along the predicted front, ``surrogate_assisted`` of
``pinneapple_design.design_optimizer``) is compared with NSGA-II calling the FEA for every individual, at equal
quality: the hypervolume of the feasible FEA-evaluated designs. Every design of the hybrid front is FEA-evaluated by
construction and re-evaluated at a finer FEA (480 gap nodes, 12 rotor angles).

    python examples/electrodynamics/12_prius_rotor_design.py            # 3 seeds, about 45 min
    python examples/electrodynamics/12_prius_rotor_design.py --quick    # 1 seed, small budgets
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machines import design as D

OUT = Path(__file__).parent / "results"


def main(quick: bool) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    seeds = (0,) if quick else (0, 1, 2)
    runs = []
    for seed in seeds:
        fea = D.rotor_evaluator(n_gap=288, n_positions=6)
        fine = D.rotor_evaluator(n_gap=480, n_positions=12, name="ipm-fea-fine")
        P = D.rotor_design_problem(fea)
        r = D.hybrid_vs_fea_only(P, n_initial=12 if quick else 24, rounds=2 if quick else 4, n_verify=4 if quick else 8,
                                 ga_budget=60 if quick else 400, surrogate_budget=600 if quick else 2000, seed=seed,
                                 verifier=fine, progress=lambda s: print(s, flush=True))
        r["seed"] = seed
        runs.append(r)
        print(f"seed {seed}: hybrid {r['hybrid']['fea_calls']} FEA calls, HV {r['hybrid']['hypervolume']:.4f}; "
              f"FEA-only reaches it after {r['fea_only']['calls_to_reach_hybrid']} calls "
              f"(final HV {r['fea_only']['hypervolume']:.4f} after {r['fea_only']['fea_calls']}); "
              f"saving {r['fea_call_saving']}", flush=True)
    res = {"runs": runs, "seconds": time.perf_counter() - t0,
           "baseline": {"W4": 0.0189, "H3": 0.0065, "H0": 0.01096, "W0": 0.042, "H1": 0.0015}}
    OUT.mkdir(exist_ok=True)
    (OUT / "12_prius_rotor_design.json").write_text(json.dumps(res, indent=1, default=float))
    _figure(runs)
    return res


def _figure(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for r in runs:
        c = r["fea_only"]["hypervolume_curve"]
        ax[0].plot(np.arange(1, len(c) + 1), c, label=f"NSGA-II on FEA, seed {r['seed']}")
        ax[0].axhline(r["hybrid"]["hypervolume"], ls="--", lw=0.8, color=ax[0].lines[-1].get_color())
        ax[0].plot([r["hybrid"]["fea_calls"]], [r["hybrid"]["hypervolume"]], "o", color=ax[0].lines[-1].get_color())
        f = r["hybrid"]["front"]
        ax[1].plot([d["magnet_mass"] for d in f], [d["torque"] for d in f], "o", label=f"hybrid front, seed {r['seed']}")
    ax[1].plot([2 * 8 * 0.0189 * 0.0065 * 0.08382 * 7500], [345.1], "k*", ms=12, label="Prius rotor")
    ax[0].set_xlabel("FEA evaluations")
    ax[0].set_ylabel("hypervolume (normalised)")
    ax[0].set_title("dots: hybrid (GP + NSGA-II) at its FEA count")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("magnet mass (kg)")
    ax[1].set_ylabel("torque at 250 A (N m, FEA)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "12_prius_rotor_design.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main("--quick" in sys.argv)
