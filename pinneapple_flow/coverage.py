"""Which PINNeAPPle capabilities are flow nodes, and why the others are not.

:func:`coverage` walks every ``pinneapple*`` package and its first-level modules, maps the registered node types onto
them (by the library calls each node declares) and attaches a curated reason (:data:`REASONS`) to whatever has no
node. ``python -m pinneapple_flow coverage -o docs/flow_studio/COVERAGE.md`` writes the table.
"""
from __future__ import annotations

import pkgutil
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# package or package.module -> why there is no node (or "covered by <node types>" when the coverage is indirect)
REASONS: dict[str, str] = {
    # compatibility shims and infrastructure
    "pinneapple_models": "shim: re-exports pinneapple_neural.architectures (covered there)",
    "pinneapple_train": "shim: re-exports pinneapple_neural.trainer (covered there)",
    "pinneapple_solvers": "shim: re-exports pinneapple_simulation.numerical_solvers (covered there)",
    "pinneapple_orchestration": "pipeline wiring for Prefect; flows are orchestrated by pinneapple_flow itself",
    "pinneapple_mcp": "the MCP server for coding agents (a protocol front end, like Flow Studio)",
    "pinneapple_flow": "this package",
    "pinneapple_registry": "artifact store and model gateway services (deployment, not a computation step)",
    "pinneapple_security": "integrity, signing, web security and audit (infrastructure used by the apps)",
    "pinneapple_core.service": "job queue / API keys of the app servers (infrastructure)",
    "pinneapple_core.workflows": "the execution engine every flow runs on",
    "pinneapple_problemdesign": "conversational problem design with an external LLM agent; the planner and "
                                "problem.generic cover the same ground offline",
    "pinneapple_quantum": "needs a quantum SDK (Qiskit / PennyLane), not installed by the flow extra",
    "pinneapple_worldmodel": "foundation-model training at GPU scale; no CPU-sized entry point",
    "pinneapple_perception": "image / video / audio inputs: no media port type in the flow schema yet",
    "pinneapple_blender": "needs Blender's bpy; viz.render uses it when installed",
    "pinneapple_twin3d": "3-D twin scene server; viz.field3d gives the in-studio 3-D view",
    "pinneapple_hub": "Hugging Face Hub transfers (network); the zoo is reachable through lib.zoo.* nodes",
    "pinneapple_catalog": "method / resource catalogue with license gates (read by the planner and docs)",
    "pinneapple_decision": "experiment-selection loop driven by an LLM backend; the planner covers routing",
    "pinneapple_arena": "YAML benchmark campaigns (long runs); single benchmarks are lib.benchmark.* nodes",
    "pinneapple_pdb": "dataset builder for benchmark publication (sharding, upload)",
    "pinneapple_veriphysics": "report layer over results; decision records + export.report cover it here",
    "pinneapple_check": "the CI physics linter (runs on files in a repository, not in a flow)",
    "pinneapple_lab": "covered by validate.lab (run, curation tiers, seal)",
    "pinneapple_llm": "LLM drafting: used by the LLM planner, problem.generic and twin.auto",
    "pinneapple_adaptation": "covered by train.finetune (transfer); meta-learning needs task families",
    "pinneapple_app": "a separate web application (not a library package)",
}


def packages() -> list[str]:
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("pinneapple")
                  and (p / "__init__.py").is_file())


def modules(pkg: str) -> list[str]:
    path = ROOT / pkg
    return sorted(f"{pkg}.{m.name}" for m in pkgutil.iter_modules([str(path)]) if not m.name.startswith("_"))


_MOD = re.compile(r"\bpp\.[a-z_]+|\bpinneapple[a-z_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def node_targets() -> dict[str, set[str]]:
    """module path -> node types that call into it."""
    from .registry import REGISTRY
    out: dict[str, set[str]] = {}
    pp = {"pp.solve": "pinneapple_physics.solving", "pp.trust_card": "pinneapple_analysis.trust",
          "pp.sobol_indices": "pinneapple_analysis.sensitivity", "pp.viz": "pinneapple_tools.visualization"}
    for s in REGISTRY.specs():
        for m in _MOD.findall(s.calls or ""):
            if m.startswith("pp."):
                out.setdefault("pinneapple", set()).add(s.type)       # the pp namespace itself
            m = pp.get(m, m)
            out.setdefault(m, set()).add(s.type)
    return out


def coverage() -> list[dict[str, str]]:
    targets = node_targets()
    rows = []
    for pkg in packages():
        mods = modules(pkg) or [pkg]
        for m in [pkg] + [x for x in mods if x != pkg]:
            nodes = sorted({t for k, v in targets.items() for t in v if k == m or k.startswith(m + ".")})
            reason = REASONS.get(m) or (REASONS.get(pkg) if m != pkg and not nodes else "")
            if m == pkg and not nodes and not reason:
                reason = ""
            rows.append({"module": m, "nodes": ", ".join(nodes), "reason": reason if not nodes else ""})
    return rows


def markdown() -> str:
    rows = coverage()
    n_nodes = sum(1 for r in rows if r["nodes"])
    n_reason = sum(1 for r in rows if not r["nodes"] and r["reason"])
    n_open = sum(1 for r in rows if not r["nodes"] and not r["reason"])
    out = ["# Flow Studio coverage of the library", "",
           "Generated by `python -m pinneapple_flow coverage` from the node registry (each node declares the library "
           "calls it wraps) and the curated reasons in `pinneapple_flow/coverage.py`.", "",
           f"{len(rows)} packages / modules: {n_nodes} with nodes, {n_reason} with a stated reason, {n_open} "
           "not reached by any node yet (any public callable can still be added as a library-call entry).", "",
           "| package / module | node types | why not a node |", "|---|---|---|"]
    for r in rows:
        out.append(f"| `{r['module']}` | {r['nodes']} | {r['reason']} |")
    return "\n".join(out) + "\n"
