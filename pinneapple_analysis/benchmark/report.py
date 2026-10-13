"""The benchmark report: tables (n in every header, integral quantities and fields apart, data cost, break-even,
a checks badge per row), the Checks, Validation, Trust card and Diagnostics sections, figures and summary.json."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_analysis.cost.equal_budget import AmortizedCost

from .checks import DID_NOT_CONVERGE

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fmt_s(x) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    if x < 1e-3:
        return f"{x * 1e6:.0f} µs"
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    return f"{x:.0f} s" if x < 1e4 else f"{x / 3600:.1f} h"


def _ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_seeds": int(len(v))}


def collect(plan) -> dict:
    """{(problem, method): {level: {seed: extracted}}} plus costs, from the run files."""
    out, cost = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(list))
    for pid, mid, prob, meth in plan.pairs():
        for lv in plan.cfg.budget_levels:
            for s in range(plan.cfg.seeds):
                tag = f"final__{pid}__{mid}__L{lv:.3f}__s{s}"
                res = {}
                if meth.per_instance:
                    for p in plan.ws.runs.glob(f"{tag}__*.json"):
                        rec = json.loads(p.read_text())
                        if rec.get("ok"):
                            res[p.stem[len(tag) + 2:]] = rec["result"]
                else:
                    rec = plan.ws.result(tag)
                    if rec and rec.get("ok"):
                        res[""] = rec["result"]
                if not res:
                    continue
                out[(pid, mid)][lv][s] = prob.extract(mid, res)
                if lv == 1.0:
                    walls = [r["train"]["wall_s"] for r in res.values() if "train" in r]
                    flops = [r["train"]["flops"] for r in res.values() if "train" in r]
                    cost[(pid, mid)]["wall"].append(float(np.mean(walls)) if walls else np.nan)
                    cost[(pid, mid)]["flops"].append(float(np.mean(flops)) if flops else np.nan)
                    t = prob.time_per_instance(mid, res, meth.per_instance) if hasattr(prob, "time_per_instance") \
                        else _time_per_instance(res, meth.per_instance)
                    cost[(pid, mid)]["infer"].append(t)
    return {"extracted": {k: dict(v) for k, v in out.items()}, "cost": {k: dict(v) for k, v in cost.items()}}


def _time_per_instance(res: dict, per_instance: bool) -> float:
    vals = []
    for r in res.values():
        inf = [v["median_s"] for k, v in r.items() if k.startswith("infer") and isinstance(v, dict) and "median_s" in v]
        t = (inf[0] if inf else 0.0) + (r["train"]["wall_s"] if per_instance else 0.0)
        vals.append(t)
    return float(np.mean(vals)) if vals else np.nan


def cell(seeds: dict, set_: str, metric: str) -> dict | None:
    per_seed, ids = [], None
    for d in seeds.values():
        e = d.get(set_, {}).get(metric)
        if e and e["values"]:
            per_seed.append(float(np.mean(e["values"])))
            ids = e.get("ids")
    m = _ms(per_seed)
    if m:
        m["n"] = len(ids) if ids is not None else None
    return m


def _fmt(c, pct=False):
    if not c:
        return "-"
    k = 100.0 if pct else 1.0
    s = f"{c['mean'] * k:.3g}" + ("%" if pct else "")
    if c["n_seeds"] > 1:
        s += f" ± {c['std'] * k:.2g}" + ("%" if pct else "")
    return s


def build_tables(plan, col, row_status: dict) -> dict:
    """Rows per problem: {method: {column key: cell, ...}} plus costs and statuses."""
    tables = {}
    for pid, prob in plan.P.items():
        rows = {}
        data_s = prob.data_cost_s(plan.ws.prepared(pid))
        per_inst = [m for m, meth in plan.M.items() if meth.per_instance and meth.supports(prob)]
        ref = None
        if per_inst and (pid, per_inst[0]) in col["cost"]:
            ref = AmortizedCost(per_query_train_s=float(np.nanmean(col["cost"][(pid, per_inst[0])]["infer"])))
        for mid, meth in plan.M.items():
            if not meth.supports(prob) or (pid, mid) not in col["extracted"]:
                continue
            seeds = col["extracted"][(pid, mid)].get(1.0, {})
            r = {"cells": {c.key: cell(seeds, c.set, c.metric) for c in prob.columns()}}
            co = col["cost"].get((pid, mid), {})
            r["train_s"] = float(np.nanmean(co["wall"])) if co.get("wall") else None
            r["flops"] = float(np.nanmean(co["flops"])) if co.get("flops") else None
            r["time_per_instance_s"] = float(np.nanmean(co["infer"])) if co.get("infer") else None
            r["data_core_s"] = None if meth.per_instance else data_s
            if ref is not None and not meth.per_instance and r["train_s"] is not None and r["time_per_instance_s"]:
                r["break_even"] = AmortizedCost(data_s=data_s or 0.0, train_s=r["train_s"],
                                                infer_s=r["time_per_instance_s"]).break_even(ref)
            main = prob.main_metric() if hasattr(prob, "main_metric") else None
            mc = cell(seeds, *main) if main else None
            r["did_not_converge"] = bool(mc and mc["mean"] >= DID_NOT_CONVERGE)
            r["status"] = row_status.get(f"{pid}/{mid}", "pass")
            if r["did_not_converge"] and r["status"].startswith("pass"):
                r["status"] = "did not converge"
            r["outputs"] = prob.method_outputs(meth)
            rows[mid] = r
        ref = prob.reference_row(plan.ws.prepared(pid)) if hasattr(prob, "reference_row") else None
        if ref:
            seeds = dict(enumerate(ref["seeds"]))
            ref = {"title": ref["title"], "cells": {c.key: cell(seeds, c.set, c.metric) for c in prob.columns()}}
        tables[pid] = {"rows": rows, "solver_time_per_instance_s": prob.solver_time_per_instance_s(plan.ws.prepared(pid)),
                       "data_core_s": data_s, "reference": ref}
    return tables


def _budget_str(b) -> str:
    parts = []
    if b.wall_s:
        parts.append(f"{b.wall_s:g} s wall clock")
    if b.flops:
        parts.append(f"{b.flops:.3g} FLOPs")
    if b.steps:
        parts.append(f"{b.steps} steps")
    return ", ".join(parts) or "unlimited"


def _badge(status: str) -> str:
    return {"pass": "✓ pass", "pass (warnings)": "✓ pass (warn)", "INVALID": "✗ INVALID",
            "did not converge": "⚠ did not converge", "checks skipped": "⚠ checks skipped"}.get(status, status)


def markdown(plan, tables, checks, validation, trust, diagnoses, banners) -> str:
    cfg = plan.cfg
    L = ["# Benchmark results", ""]
    for b in banners:
        L += [f"> **{b}**", ""]
    hw = json.loads((plan.ws.root / "hardware.json").read_text()) if (plan.ws.root / "hardware.json").exists() else {}
    L += [f"Hardware: {hw.get('cpu')} ({hw.get('logical_cpus')} logical CPUs), GPU: {hw.get('gpu')}; torch "
          f"{hw.get('torch')}. Every training run uses {cfg.threads_per_run} thread(s); {cfg.n_workers} run(s) at a "
          f"time. Seeds: {cfg.seeds}; search: {cfg.n_trials} trials per method at {cfg.search_fraction:.3g} x the "
          "budget, validation data only. Cells: mean ± std over seeds of the per-seed mean over the n instances of the "
          "column. A row whose error is no better than a trivial prediction is labelled 'did not converge'; one that "
          "failed a gate is INVALID.", ""]
    for pid, t in tables.items():
        prob = plan.P[pid]
        b = cfg.budget_for(pid)
        L += [f"## {prob.title} (budget per run: {_budget_str(b)})", ""]
        cols = prob.columns()
        for kind, head in (("field", "Fields"), ("integral", "Integral quantities")):
            cs = [c for c in cols if c.kind == kind]
            if not cs:
                continue
            L += [f"### {head}", "", "| Method | " + " | ".join(c.label for c in cs) + " | Checks |",
                  "|---|" + "---|" * (len(cs) + 1)]
            for mid, r in t["rows"].items():
                meth = plan.M[mid]
                outs = prob.method_outputs(meth)
                derived = any(r["cells"].get(c.key) for c in cs)   # e.g. an integral computed from the predicted field
                if not derived and ((kind == "field" and outs == "integrals") or (kind == "integral" and outs == "fields")):
                    vals = ["integrals only" if kind == "field" else "fields only"] * len(cs)
                else:
                    vals = []
                    for c in cs:
                        v = r["cells"].get(c.key)
                        s = _fmt(v, c.pct)
                        expected = _n_in_label(c.label)
                        if v and v.get("n") is not None and expected is not None and v["n"] != expected:
                            s += f" [n={v['n']}]"
                        vals.append(s)
                L.append(f"| {prob.method_title(meth)} | " + " | ".join(vals) + f" | {_badge(r['status'])} |")
            if t.get("reference"):
                L.append(f"| {t['reference']['title']} | " + " | ".join(_fmt(t["reference"]["cells"].get(c.key), c.pct)
                                                                       for c in cs) + " | reference |")
            L.append("")
        L += ["### Cost", "", "| Method | Training (s) | Training FLOPs | Simulation data (core-s) | Time per new instance "
              "| Break-even vs per-instance (queries) |", "|---|---|---|---|---|---|"]
        for mid, r in t["rows"].items():
            be = r.get("break_even")
            L.append(f"| {prob.method_title(plan.M[mid])} | {fmt_s(r['train_s'])} | {r['flops']:.2e} | "
                     f"{fmt_s(r['data_core_s']) if r['data_core_s'] else '-'} | {fmt_s(r['time_per_instance_s'])} | "
                     f"{'-' if be is None else f'{be:.1f}'} |" if r["flops"] else
                     f"| {prob.method_title(plan.M[mid])} | {fmt_s(r['train_s'])} | - | - | {fmt_s(r['time_per_instance_s'])} | - |")
        if t["solver_time_per_instance_s"]:
            L.append(f"| Reference solver | - | - | - | {fmt_s(t['solver_time_per_instance_s'])} | - |")
        L.append("")
    L += ["## Diagnostics and suggestions", ""]
    if diagnoses:
        L += [f"- {d.line()}" for d in sorted(diagnoses, key=lambda d: ("error", "warning", "info").index(d.severity))]
    else:
        L.append("No issue found in the measured evidence.")
    L += ["", "## Validation", ""]
    for pid, v in validation.items():
        L += [f"### {plan.P[pid].title}: prepared artifacts {v.get('status', '?')}", "", "| Check | Status | Value | Detail |",
              "|---|---|---|---|"]
        for c in v.get("checks", []) + v.get("validation", []):
            L.append(f"| {c['name']} | {_status(c)} | {_short(c.get('value'))} | {_short(c.get('detail'), 140)} |")
        L.append("")
    L += ["## Trust card", ""]
    for pid, card in trust.items():
        L += [f"### {plan.P[pid].title}", "", "| Stage | Question | Status | Evidence |", "|---|---|---|---|"]
        for _k, s in card.items():
            L.append(f"| {s['title']} | {s['question']} | {s['status']} | {_short('; '.join(s['evidence']), 160)} |")
        L.append("")
    L += ["## Checks", "", "| Stage | Subject | Check | Status | Value | Threshold |", "|---|---|---|---|---|---|"]
    for c in checks:
        d = c if isinstance(c, dict) else c.to_dict()
        L.append(f"| {d['stage']} | {d['subject']} | {d['name']} | {_status(d)} | {_short(d.get('value'))} | "
                 f"{_short(d.get('threshold'))} |")
    return "\n".join(L) + "\n"


def _n_in_label(label: str):
    import re
    m = re.search(r"n = (\d+)", label)
    return int(m.group(1)) if m else None


def _status(c: dict) -> str:
    p = c.get("passed")
    return "n/a" if p is None else ("pass" if p else ("FAIL" if c.get("severity", "error") == "error" else "warn"))


def _short(x, n=60) -> str:
    s = "" if x is None else (json.dumps(x) if isinstance(x, (dict, list)) else str(x))
    s = s.replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def write_csv(path: Path, plan, tables) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "method", "column", "kind", "set", "metric", "n", "mean", "std", "n_seeds", "status"])
        for pid, t in tables.items():
            cols = {c.key: c for c in plan.P[pid].columns()}
            for mid, r in t["rows"].items():
                for k, v in r["cells"].items():
                    if v:
                        c = cols[k]
                        w.writerow([pid, mid, c.label, c.kind, c.set, c.metric, v.get("n"), v["mean"], v["std"],
                                    v["n_seeds"], r["status"]])
                for k in ("train_s", "flops", "data_core_s", "time_per_instance_s", "break_even"):
                    if r.get(k) is not None:
                        w.writerow([pid, mid, k, "cost", "", k, "", r[k], "", "", r["status"]])


def _style(ax):
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def figures(plan, col, tables, fig_dir: Path, banners) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    mids = list(plan.M)
    color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(mids)}
    mark = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(mids)}
    pids = list(plan.P)
    files = []
    # error vs budget
    fig, axes = plt.subplots(1, len(pids), figsize=(4.2 * len(pids), 3.6), squeeze=False)
    for ax, pid in zip(axes[0], pids, strict=False):
        prob = plan.P[pid]
        main = prob.main_metric() if hasattr(prob, "main_metric") else None
        for mid in mids:
            lv = col["extracted"].get((pid, mid))
            if not lv or not main:
                continue
            xs, ys, es = [], [], []
            for L in sorted(lv):
                c = cell(lv[L], *main)
                if c:
                    xs.append(L * (plan.cfg.budget_for(pid).wall_s or 1)), ys.append(c["mean"]), es.append(c["std"])
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=color[mid], marker=mark[mid], lw=2, capsize=3,
                            label=prob.method_title(plan.M[mid]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(prob.title, fontsize=10)
        ax.set_xlabel("training budget (s)", fontsize=8)
        ax.set_ylabel(f"{main[1]} on {main[0]}" if main else "", fontsize=8)
        _style(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Error vs training budget (mean ± std over seeds)" + (f" — {'; '.join(banners)}" if banners else ""),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "error_vs_budget.png", dpi=150)
    plt.close(fig)
    files.append("error_vs_budget.png")
    # Pareto
    fig, axes = plt.subplots(1, len(pids), figsize=(4.2 * len(pids), 3.6), squeeze=False)
    for ax, pid in zip(axes[0], pids, strict=False):
        prob = plan.P[pid]
        main = prob.main_metric() if hasattr(prob, "main_metric") else None
        for mid, r in tables[pid]["rows"].items():
            c = cell(col["extracted"][(pid, mid)].get(1.0, {}), *main) if main else None
            if c and r["time_per_instance_s"]:
                ax.errorbar([r["time_per_instance_s"]], [c["mean"]], yerr=[c["std"]], color=color[mid], marker=mark[mid],
                            ms=8, lw=0, elinewidth=1.5, capsize=3, label=prob.method_title(plan.M[mid]))
        st = tables[pid]["solver_time_per_instance_s"]
        if st:
            ax.axvline(st, color=INK2, ls="--", lw=1)
            ax.text(st, 0.98, " solver", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(prob.title, fontsize=10)
        ax.set_xlabel("time per new instance (s)", fontsize=8)
        _style(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Error vs time per new instance (lower left is better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "pareto_error_vs_inference.png", dpi=150)
    plt.close(fig)
    files.append("pareto_error_vs_inference.png")
    # summary table
    lines = []
    for pid in pids:
        prob = plan.P[pid]
        for label, kind, main, ood, pct in summary_metrics(prob):
            for mid, r in tables[pid]["rows"].items():
                outs = prob.method_outputs(plan.M[mid])
                if (kind == "field" and outs == "integrals") or (kind == "integral" and outs == "fields"):
                    continue
                lv = col["extracted"][(pid, mid)].get(1.0, {})
                c = cell(lv, *main) if main else None
                o = None
                for cand in ([ood] if ood and isinstance(ood[0], str) else (ood or [])):   # first set this method has
                    o = cell(lv, *cand)
                    if o:
                        break
                f = (lambda v: f"{100 * v:.3g}%") if pct else (lambda v: f"{v:.3g}")
                lines.append([prob.title + (f", {label}" if label else ""), prob.method_title(plan.M[mid]),
                              (f"{f(c['mean'])} (n={c['n']})" if c else "-"), (f"{f(o['mean'])} (n={o['n']})" if o else "-"),
                              fmt_s(r["time_per_instance_s"]), _badge(r["status"])])
    h = 0.27 * (len(lines) + 1) + 1.2 + 0.25 * len(banners)
    fig = plt.figure(figsize=(12, h))
    ax = fig.add_axes([0.02, 0.45 / h, 0.96, 1 - (1.0 + 0.25 * len(banners)) / h])
    ax.axis("off")
    tab = ax.table(cellText=lines or [["-"] * 6], colLabels=["Problem", "Method", "Error, in distribution",
                                                              "Error, OOD", "Time per new instance", "Checks"],
                   cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1], colWidths=[0.26, 0.2, 0.15, 0.15, 0.11, 0.13])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    for (i, _j), cl in tab.get_celld().items():
        cl.set_edgecolor(GRID)
        if i == 0:
            cl.set_text_props(weight="bold", color=INK)
    fig.text(0.02, 1 - 0.12 / h, "Equal-budget benchmark (pinneapple_analysis.benchmark)", fontsize=11, weight="bold",
             va="top")
    for k, b in enumerate(banners):
        fig.text(0.02, 1 - (0.42 + 0.25 * k) / h, b, fontsize=8.5, color="#e34948", va="top", weight="bold")
    fig.text(0.02, 0.12 / h, "Errors: each problem's main metric (relative unless stated); n = instances; mean over "
                              "seeds. Per-instance methods' time = their training.", fontsize=7, color=INK2)
    fig.savefig(fig_dir / "summary_table.png", dpi=200)
    plt.close(fig)
    files.append("summary_table.png")
    return files


def summary_metrics(prob):
    """(label, kind, (set, metric), (ood set, metric) or None, pct) lines of the summary figure for a problem."""
    if hasattr(prob, "summary_metrics"):
        return prob.summary_metrics()
    main = prob.main_metric() if hasattr(prob, "main_metric") else None
    ood = prob.ood_metric() if hasattr(prob, "ood_metric") else None
    return [("", "field", main, ood, False)]


def to_dataframe(plan, tables):
    import pandas as pd
    rows = []
    for pid, t in tables.items():
        cols = {c.key: c for c in plan.P[pid].columns()}
        for mid, r in t["rows"].items():
            d: dict[str, Any] = {"problem": pid, "method": mid, "status": r["status"]}
            for k, v in r["cells"].items():
                d[cols[k].label] = None if v is None else v["mean"]
            d.update({k: r.get(k) for k in ("train_s", "flops", "data_core_s", "time_per_instance_s", "break_even")})
            rows.append(d)
    return pd.DataFrame(rows)
