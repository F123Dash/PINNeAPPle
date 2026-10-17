"""What the zoo's demo Space shows: a model's prediction for one input and along a sweep of one input, each with
its physics checks and validity range.

The Space (``apps/zoo_space/app.py``, published to ``huggingface.co/spaces/PINNeAPPle-Labs/zoo-demo`` by
``scripts/zoo/publish_to_hub.py``) is only the UI around these functions; the same calls work in a notebook:

    from pinneapple_hub.zoo import demo
    c = demo.controls("airfoil_cst_rans_mlp")                      # inputs, units, ranges, a held-out case
    r = demo.sweep("airfoil_cst_rans_mlp", {k["name"]: k["value"] for k in c}, vary="alpha_deg")
    demo.figure(r).savefig("polar.png"); print(demo.report(r))

Out-of-range parts of a sweep are shaded red (the query checks fail: an extrapolation), warnings yellow.
"""
from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _artifacts as art

__all__ = ["models", "controls", "sweep", "figure", "report", "trust"]

STATUS_COLOR = {"pass": "#2e7d32", "warn": "#f9a825", "fail": "#c62828"}


def models() -> list[str]:
    """Names of the zoo models the demo offers."""
    return sorted(art.manifest()["models"])


@functools.cache
def _model(name: str):
    """The model, with its reference data fetched once (the query checks read it offline afterwards)."""
    from . import evaluate, load
    zm = load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluate(zm)
    return zm


@functools.cache
def trust(name: str) -> dict[str, Any]:
    """The model's trust report as a whole (``pp.zoo.validate``): decision and the reasons."""
    from . import validate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = validate(_model(name))
    return {"decision": r.decisao, "reasons": list(getattr(r, "motivos", []) or [])}


def _range(zm, k: str, X: np.ndarray, j: int) -> tuple[float, float]:
    b = zm.bounds()
    if k in b:
        return float(b[k][0]), float(b[k][1])
    return float(X[:, j].min()), float(X[:, j].max())          # no box bound (derived limits): the held-out span


@functools.cache
def _controls(name: str) -> tuple:
    zm = _model(name)
    X, _ = zm.reference("test")
    out = []
    for j, d in enumerate(zm.card.inputs):
        lo, hi = _range(zm, d["name"], X, j)
        out.append({"name": d["name"], "unit": d.get("unit", ""), "description": d.get("description", ""),
                    "lo": lo, "hi": hi, "value": float(f"{X[0, j]:.6g}")})
    return tuple(out)


def controls(name: str) -> list[dict[str, Any]]:
    """One entry per input of the model: name, unit, description, range (validity box, or the held-out span for
    inputs bounded only through derived quantities) and a starting value (the first held-out test case)."""
    return [dict(c) for c in _controls(name)]


def _checks(zm, X: np.ndarray) -> list[dict[str, Any]]:
    from .checks import has_query_checks, query_checks
    if has_query_checks(zm):
        return query_checks(zm, X)
    inside = zm.in_range(X)
    return [{"checks": [{"name": "inputs inside the validity range", "status": "pass" if ok else "fail",
                         "rule": "PC601", "kind": "trust", "detail_text": ""}],
             "verdict": {"status": "pass" if ok else "fail"}} for ok in inside]


def sweep(name: str, values: Mapping[str, float], vary: str, *, n: int = 61, margin: float = 0.15) -> dict[str, Any]:
    """Predictions and query checks at ``values`` and along ``vary`` from ``margin`` below to ``margin`` above its
    range (so the sweep shows where the model stops being trusted)."""
    zm = _model(name)
    names = zm.input_names
    if vary not in names:
        raise KeyError(f"{name} has no input {vary!r}; inputs: {names}")
    c = {d["name"]: d for d in _controls(name)}
    lo, hi = c[vary]["lo"], c[vary]["hi"]
    a, b = lo - margin * (hi - lo), hi + margin * (hi - lo)
    if lo > 0:
        a = max(a, lo / 2)                     # positive physical inputs (log features) stay positive
    x0 = np.array([float(values[k]) for k in names])
    grid = np.linspace(a, b, n)
    X = np.repeat(x0[None], n, 0)
    X[:, names.index(vary)] = grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = zm.predict(X, warn=False)
        point = zm.predict(x0[None], warn=False)
    return {"model": name, "vary": vary, "x": grid, "range": (lo, hi), "inputs": dict(zip(names, x0.tolist(), strict=True)),
            "pred": {k: np.asarray(v, float) for k, v in pred.items()},
            "status": [r["verdict"]["status"] for r in _checks(zm, X)],
            "point": {k: float(np.asarray(v).ravel()[0]) for k, v in point.items()},
            "point_checks": _checks(zm, x0[None])[0], "units": {d["name"]: d.get("unit", "") for d in
                                                                [*zm.card.inputs, *zm.card.outputs]}}


def _main_outputs(pred: Mapping[str, np.ndarray]) -> list[str]:
    return [k for k in pred if not k.endswith("_std") and "upper95" not in k]


def figure(result: Mapping[str, Any]):
    """Matplotlib figure of a :func:`sweep`: one panel per output (mean, +-2 std band of the ensemble or the
    conformal 95 % upper bound when the model has one), the chosen input marked, the checks' status shaded."""
    from matplotlib.figure import Figure
    outs = _main_outputs(result["pred"])
    fig = Figure(figsize=(4.2 * len(outs), 3.6))                  # not pyplot: nothing is kept open per request
    axes = fig.subplots(1, len(outs), squeeze=False)
    x, st, units = result["x"], result["status"], result["units"]
    for ax, k in zip(axes[0], outs, strict=True):
        y = result["pred"][k]
        ax.plot(x, y, color="#1f4e79", lw=2, label=k)
        if f"{k}_std" in result["pred"]:
            s = result["pred"][f"{k}_std"]
            ax.fill_between(x, y - 2 * s, y + 2 * s, color="#1f4e79", alpha=0.18, lw=0, label="±2 std (ensemble)")
        up = f"{k.removesuffix('_k_w')}_upper95_k_w" if k.endswith("_k_w") else ""
        if up in result["pred"]:
            ax.plot(x, result["pred"][up], color="#1f4e79", ls="--", lw=1, label="95 % upper bound")
        for i in range(len(x) - 1):
            if st[i] != "pass" or st[i + 1] != "pass":
                worst = "fail" if "fail" in (st[i], st[i + 1]) else "warn"
                ax.axvspan(x[i], x[i + 1], color=STATUS_COLOR[worst], alpha=0.12, lw=0)
        x0 = result["inputs"][result["vary"]]
        ax.axvline(x0, color="k", lw=0.8, ls=":")
        ax.plot([x0], [result["point"][k]], "o", color="k", ms=5)
        ax.set_xlabel(f"{result['vary']} [{units.get(result['vary'], '')}]")
        ax.set_ylabel(f"{k} [{units.get(k, '')}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{result['model']}: shaded = checks warn (yellow) / fail (red, outside the validity range)",
                 fontsize=9)
    fig.tight_layout()
    return fig


def report(result: Mapping[str, Any]) -> str:
    """Markdown: the prediction at the chosen input, the verdict and every query check, and the model's trust
    decision as a whole."""
    name = result["model"]
    e = art.entry("models", name)
    pc = result["point_checks"]
    units = result["units"]
    t = trust(name)
    lines = [f"### {name} v{e['version']}", "",
             f"**Verdict for this input: {pc['verdict']['status'].upper()}**"
             + (f" ({pc['verdict'].get('title', '')})" if pc["verdict"].get("title") else ""), "",
             "| Output | Value | Unit |", "|---|---|---|",
             *(f"| `{k}` | {v:.6g} | {units.get(k, '')} |" for k, v in result["point"].items()), "",
             "| Check | Rule | Status | Detail |", "|---|---|---|---|",
             *(f"| {c['name']} | {c.get('rule', '')} | {c['status']} | {c.get('detail_text', '')} |"
               for c in pc["checks"]), "",
             f"Model trust decision (`pp.zoo.validate`: card metrics reproduced, physics checks): "
             f"**{t['decision']}**", "",
             f"Model card: https://huggingface.co/{e['hf_repo']} · revision `{e['revision']}`"]
    return "\n".join(lines)
