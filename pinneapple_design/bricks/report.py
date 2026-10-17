"""Summaries of a brick experiment: sensitivity to the joint model, the strategy comparison chart, a Markdown report."""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from .goals import Goal
from .parts import JointModel

SERIES = {"evolution": "#2a78d6", "surrogate": "#eb6834"}     # categorical slots 1 and 2 of a validated palette
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def sensitivity(goal: Goal, asm, factors: Sequence[float] | None = None, shear_caps=(None, 1.0, 0.3)) -> list[dict]:
    """Re-evaluate ``asm`` with the clutch scaled over its assumed band and with capped stud shear."""
    base = goal.joint
    lo, hi = base.clutch_band
    factors = factors or (lo, 0.75, 1.0, 1.25, hi)
    rows = []
    for f in factors:
        for cap in shear_caps:
            j = JointModel(clutch_n=base.clutch_n * f, clutch_band=base.clutch_band, mu=base.mu, shear_n=cap)
            g = type(goal).__new__(type(goal))
            g.__dict__.update(goal.__dict__)
            g.joint = j
            ev = g.evaluate(asm)
            rows.append({"clutch_n": j.clutch_n, "clutch_factor": f, "shear_cap_n": cap, "feasible": ev.feasible,
                         "status": ev.status, "explanation": ev.explanation,
                         "load_factor": ev.metrics.get("load_factor"), "utilisation": ev.metrics.get("utilisation")})
    return rows


def plot_comparison(summaries: dict[str, dict], path: str, *, ylabels: dict[str, str] | None = None) -> str:
    """Best feasible score versus physics evaluations: one panel per goal, median line (seeds without a feasible
    build count as worst) and min-max band of the feasible seeds per strategy, with the feasible count at the end."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    goals = list(summaries)
    fig, axes = plt.subplots(1, len(goals), figsize=(4.2 * len(goals), 3.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, goals, strict=True):
        ax.set_facecolor(SURFACE)
        ends = []
        for s, summ in summaries[g].items():
            C = np.asarray(summ["curves"], float)
            x = np.arange(1, C.shape[1] + 1)
            ok = np.isfinite(C)
            # a seed with no feasible build yet ranks below every feasible one: the median exists only where more
            # than half of the seeds are feasible, the band spans the feasible seeds
            med = np.median(np.where(ok, C, -np.inf), 0)
            med[~np.isfinite(med)] = np.nan
            with warnings.catch_warnings():                 # columns where no seed is feasible yet
                warnings.simplefilter("ignore", RuntimeWarning)
                lo = np.where(ok.any(0), np.nanmin(np.where(ok, C, np.nan), 0), np.nan)
                hi = np.where(ok.any(0), np.nanmax(np.where(ok, C, np.nan), 0), np.nan)
            col = SERIES.get(s, MUTED)
            ax.fill_between(x, lo, hi, color=col, alpha=0.16, linewidth=0)
            n_ok = int(ok[:, -1].sum())
            ax.plot(x, med, color=col, lw=2, label=f"{s}: median and range of {C.shape[0]} seeds")
            last = np.where(np.isfinite(med))[0]
            if len(last):
                ends.append([med[last[-1]], x[last[-1]], f"{med[last[-1]]:.0f} ({n_ok}/{C.shape[0]} feasible)"])
        ends.sort()
        span = np.ptp(ax.get_ylim())
        for k in range(1, len(ends)):                     # keep the end labels apart
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.07 * span)
        for yv, xv, txt in ends:
            ax.annotate(txt, (xv, yv), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK)
        ax.set_title(g, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("physics evaluations", fontsize=9, color=MUTED)
        ax.set_ylabel((ylabels or {}).get(g, "best score"), fontsize=9, color=MUTED)
        ax.grid(True, color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c2")
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def write_report(path: str, *, validation: dict, comparisons: dict[str, dict], best: dict[str, dict],
                 sensitivities: dict[str, list[dict]], references: dict | None = None, files: dict | None = None,
                 joint: JointModel | None = None, cross_checks: dict[str, dict] | None = None,
                 notes: Sequence[str] = ()) -> str:
    """Markdown report: what was measured, the closed-form checks (and where they disagree), the strategy comparison
    with its spread over seeds, the best builds, the sensitivity to the joint model and its limits."""
    joint = joint or JointModel()
    L = ["# Self-modifying interlocking-brick builds", "",
         "Every candidate build is checked by a static equilibrium analysis (rigid parts, stud clutch, friction); "
         "rejected candidates are explained (which joint let go, or where the resultant left its support) and "
         "replayed with the Newton rigid-body engine.", "",
         "## Joint model", "",
         f"- Clutch per stud: {joint.clutch_n} N. Source: {joint.source}.",
         f"- Uncertainty: none published; the sensitivity study scales it by {joint.clutch_band[0]}x to "
         f"{joint.clutch_band[1]}x (an assumed band).",
         f"- Friction on smooth contacts: mu = {joint.mu} (assumed).",
         "- Limits: " + "; ".join(joint.notes) + ".", "",
         "## Checks against closed forms", "",
         f"Trust report decision: **{validation['report'].decisao}** "
         f"(score {validation['report'].score:.2f}, coverage {validation['report'].cobertura:.2f}).", "",
         "| check | simulated | closed form | rel. error | tolerance | pass | note |", "|---|---|---|---|---|---|---|"]
    for r in validation["rows"]:
        L.append(f"| {r['check']} | {_fmt(r['simulated'], 6)} {r['unit']} | {_fmt(r['reference'], 6)} | "
                 f"{r['rel_error']:.2e} | {r['tolerance']} | {'yes' if r['passed'] else '**no**'} | {r['note']} |")
    if validation.get("harmonic_newton"):
        L += ["", "Newton on the harmonic stack (offsets scaled by a factor; the closed form stands for factor <= 1):",
              "", "| factor | top-block drop [mm] | Newton | closed form | agree |", "|---|---|---|---|---|"]
        for r in validation["harmonic_newton"]:
            L.append(f"| {r['factor']} | {r['newton_top_drop_mm']:.2f} | {'stands' if r['newton_stands'] else 'falls'} "
                     f"| {'stands' if r['closed_form_stands'] else 'falls'} | {'yes' if r['agree'] else '**no**'} |")
    if cross_checks:
        L += ["", "## Statics against Newton on searched candidates", "",
              "Candidates the statics calls stable (nothing should move) or a mechanism (something should), replayed "
              "by Newton with every stud joint intact under the goal's loads. Joint failures are left out: Newton has "
              "no clutch model to judge them.", "",
              "| goal | stable, stays | stable, moves | mechanism, moves | mechanism, stays | agreement | dt [s] |",
              "|---|---|---|---|---|---|---|"]
        for g, cc in cross_checks.items():
            c = cc["confusion"]
            L.append(f"| {g} | {c['stable_and_stays']} | {c['stable_but_moves']} | {c['tips_and_moves']} | "
                     f"{c['tips_but_stays']} | {cc['agreement']:.0%} | {cc['dt']:g} |")
    if notes:
        L += ["", "Where the simulations disagree with each other or with the closed forms:", ""] + [f"- {n}" for n in notes]
    L += ["", "## Search strategies at equal evaluation budget", ""]
    for g, comp in comparisons.items():
        L += [f"### {g}", "", "| strategy | best (mean +- std) | min | max | feasible runs | per seed |",
              "|---|---|---|---|---|---|"]
        for s, summ in comp.items():
            per = ", ".join("-" if not np.isfinite(v) else f"{v:.1f}" for v in summ["per_seed"])
            L.append(f"| {s} | {summ['best_mean']:.1f} +- {summ['best_std']:.1f} | {summ['best_min']:.1f} | "
                     f"{summ['best_max']:.1f} | {summ['success_rate']:.0%} | {per} |")
        L.append(f"\nBudget {next(iter(comp.values()))['budget']} physics evaluations per run, seeds "
                 f"{next(iter(comp.values()))['seeds']}.")
        if g in best:
            b = best[g]
            L += ["", f"Best build ({b['strategy']}, seed {b['seed']}): {b['n_bricks']} parts, score "
                      f"{b['score']:.1f}. {b['explanation']}"]
        if references and g in references:
            L.append("Closed-form references: " + ", ".join(f"{k} {v:.1f} mm" for k, v in references[g].items()) + ".")
        if g in sensitivities:
            L += ["", "Sensitivity of the best build to the joint model:", "",
                  "| clutch [N] | shear cap [N/stud] | result | load factor | utilisation |", "|---|---|---|---|---|"]
            for r in sensitivities[g]:
                L.append(f"| {r['clutch_n']:.2f} | {_fmt(r['shear_cap_n'])} | "
                         f"{'stands' if r['feasible'] else r['status']} | {_fmt(r['load_factor'])} | "
                         f"{_fmt(r['utilisation'])} |")
        L.append("")
    if files:
        import os
        here = os.path.dirname(os.path.abspath(path))
        L += ["## Files", ""] + [f"- {k}: `{os.path.relpath(v, here)}`" for k, v in files.items()] + [""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path
