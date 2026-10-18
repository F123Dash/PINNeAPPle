"""Figures of a forecasting study (matplotlib, static PNG).

Families keep one colour everywhere (fixed order, never re-assigned when a family is missing from a panel); the
baseline is drawn as a neutral grey reference line at skill 0.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["FAMILY_COLORS", "plot_skill_vs_horizon", "plot_overfitting", "plot_surrogate_tests", "plot_pit",
           "plot_fan_charts"]

FAMILY_COLORS = {"data": "#2a78d6", "data+physics": "#eb6834", "physical": "#1baf7a", "hybrid": "#6250d6",
                 "baseline": "#52514e"}
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_skill_vs_horizon(results: Sequence[dict], path: str, *, lyapunov: dict[str, float] | None = None,
                          ncols: int = 4, floor: float = -0.5, titles: dict[str, str] | None = None) -> str:
    """One panel per series: RMSE skill vs persistence of each family's representative (chosen on the validation
    block) against the horizon; filled markers where the Holm-adjusted Diebold-Mariano test against persistence is
    significant at 5 %. The horizon is in Lyapunov times when ``lyapunov[series]`` (exponent per unit time) is
    given. Skill below ``floor`` is drawn at the floor."""
    plt = _plt()
    n = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), squeeze=False)
    for ax, res in zip(axes.ravel(), results, strict=False):
        _style(ax)
        H = np.asarray(res["protocol"]["horizons"], float)
        lam = (lyapunov or {}).get(res["series"])
        x = H * res["dt"] * lam if lam else H
        ax.axhline(0, color=FAMILY_COLORS["baseline"], lw=1.2, ls="--")
        for fam, name in sorted(res.get("family_representative", {}).items(),
                                key=lambda kv: list(FAMILY_COLORS).index(kv[0])):
            if fam == "baseline":
                continue
            m = res["models"][name]
            sk = np.maximum(np.asarray(m["test"]["skill_rmse"], float), floor)
            p = np.array([(d or {}).get("p_holm", np.nan) for d in m["dm_vs_persistence"]], float)
            c = FAMILY_COLORS[fam]
            ax.plot(x, sk, color=c, lw=2, label=f"{fam}: {name}")
            sig = p < 0.05
            ax.scatter(x[sig], sk[sig], color=c, s=22, zorder=3)
            ax.scatter(x[~sig], sk[~sig], facecolor="white", edgecolor=c, s=22, zorder=3, lw=1.2)
        ax.set_xscale("log")
        ax.set_ylim(floor - 0.05, 1.02)
        ax.set_title((titles or {}).get(res["series"], res["series"]), fontsize=9, color=_INK)
        ax.set_xlabel("horizon [Lyapunov times]" if lam else "horizon [samples]", fontsize=8, color=_MUTED)
        ax.set_ylabel("RMSE skill vs persistence", fontsize=8, color=_MUTED)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_overfitting(studies: Sequence[dict], path: str) -> str:
    """Per series: train, validation and test error of every training variant (bars), and the cross-validation
    estimates of each scheme (markers) for the variants where they were computed."""
    plt = _plt()
    fig, axes = plt.subplots(len(studies), 1, figsize=(10, 3.2 * len(studies)), squeeze=False)
    cols = {"train": "#9ec5f0", "val": "#2a78d6", "test": "#0b3d7a"}
    marks = {"shuffled_kfold": ("v", "#e34948"), "walk_forward_no_purge": ("s", "#eda100"),
             "purged_walk_forward": ("o", "#1baf7a"), "purged_kfold": ("D", "#6250d6")}
    for ax, st in zip(axes[:, 0], studies, strict=False):
        _style(ax)
        names = list(st["variants"])
        xs = np.arange(len(names))
        w = 0.26
        for k, (key, c) in enumerate(cols.items()):
            ax.bar(xs + (k - 1) * w, [st["variants"][n][key] for n in names], w * 0.92, color=c, label=key)
        for sch, (mk, c) in marks.items():
            pts = [(i, st["variants"][n]["cv"][sch]["estimate"]) for i, n in enumerate(names)
                   if sch in st["variants"][n]["cv"]]
            if pts:
                i, v = zip(*pts, strict=False)
                ax.scatter(np.asarray(i) + w, v, marker=mk, color=c, edgecolor="white", s=46, zorder=4,
                           label=f"CV: {sch.replace('_', ' ')}")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("mean RMSE / std", fontsize=8, color=_MUTED)
        ax.set_title(st["series"], fontsize=9, color=_INK)
        ax.legend(fontsize=6, frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_surrogate_tests(reports: dict[str, dict], path: str, *, ncols: int = 4) -> str:
    """Nonlinear prediction error of each series (vertical line) against its shuffled and IAAFT surrogates."""
    plt = _plt()
    names = list(reports)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        _style(ax)
        t = reports[name]["surrogate_tests"]
        for kind, c in (("shuffle", "#e34948"), ("iaaft", "#2a78d6")):
            s = np.asarray(t[f"npe_{kind}"]["surrogates"])
            ax.hist(s, bins=15, color=c, alpha=0.55, label=f"{kind} (p={t[f'npe_{kind}']['p_value']:.2f})")
        ax.axvline(t["npe_iaaft"]["observed"], color=_INK, lw=2, label="series")
        ax.set_title(name, fontsize=9, color=_INK)
        ax.set_xlabel("NN prediction error / std (h=1)", fontsize=8, color=_MUTED)
        ax.legend(fontsize=6, frameon=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_pit(entries: list[dict], path: str, *, ncols: int = 4) -> str:
    """PIT histograms. ``entries``: dicts with ``title``, ``freq`` (relative frequencies) and ``deviation``."""
    plt = _plt()
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), squeeze=False)
    for ax, e in zip(axes.ravel(), entries, strict=False):
        _style(ax)
        f = np.asarray(e["freq"])
        nb = len(f)
        ax.bar((np.arange(nb) + 0.5) / nb, f * nb, 0.92 / nb, color=FAMILY_COLORS.get(e.get("family"), "#2a78d6"))
        ax.axhline(1, color=_MUTED, ls="--", lw=1)
        ax.set_ylim(0, max(2.5, float(f.max() * nb) * 1.1))
        ax.set_title(f"{e['title']}\ndeviation {e['deviation']:.2f}", fontsize=7, color=_INK)
        ax.set_xlabel("PIT", fontsize=7, color=_MUTED)
    for ax in axes.ravel()[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_fan_charts(panels: list[dict], path: str, *, ncols: int = 2) -> str:
    """Forecast fans on the test period. Each panel: ``title``, ``t_hist``/``y_hist`` (observed before the origin),
    ``t_future``/``y_future`` (what happened), ``horizons_t`` and ``quantiles`` (5 x n_h: 5, 25, 50, 75, 95 %)."""
    plt = _plt()
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.5 * nrows), squeeze=False)
    for ax, p in zip(axes.ravel(), panels, strict=False):
        _style(ax)
        c = FAMILY_COLORS.get(p.get("family"), "#2a78d6")
        q = np.asarray(p["quantiles"])
        ht = np.asarray(p["horizons_t"])
        ax.fill_between(ht, q[0], q[4], color=c, alpha=0.18, lw=0, label="90 % interval")
        ax.fill_between(ht, q[1], q[3], color=c, alpha=0.35, lw=0, label="50 % interval")
        ax.plot(ht, q[2], color=c, lw=2, label="median")
        ax.plot(p["t_hist"], p["y_hist"], color=_MUTED, lw=1.2, label="observed")
        ax.plot(p["t_future"], p["y_future"], color=_INK, lw=1.2, ls=":", label="truth (test)")
        ax.set_title(p["title"], fontsize=8, color=_INK)
        ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
