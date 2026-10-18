# The flow schema (`pinneapple-flow/1`) and how another app reuses it

`pinneapple_flow` is the library behind Flow Studio. Its document format, port types, node registry, runtime and
store are independent of the GUI, so another application (for example the PINNeAPPle-CFD app) can depend on the
package, register its own node types and run or store the same documents.

```text
pip install "pinneapple[flow]"            # or: pip install -e ".[flow]" from a checkout
python -m pinneapple_flow schema          # the JSON Schema below, generated from the code
python -m pinneapple_flow catalog --json  # every node type: ports, parameter schema, availability
```

## The document

```json
{
  "schema": "pinneapple-flow/1",
  "id": "heat-rod",
  "name": "Heat in a rod",
  "description": "free text",
  "nodes": [
    {"id": "geo", "type": "geometry.domain", "params": {"dimension": 1}, "position": {"x": 0, "y": 0}, "label": "Rod"},
    {"id": "eq",  "type": "equations.system", "params": {"template": "heat"}}
  ],
  "edges": [
    {"id": "geo.geometry->eq.geometry", "source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"}
  ],
  "meta": {"planner": "heuristic", "prompt": "heat in a rod"}
}
```

| Field | Rules |
|---|---|
| `schema` | exactly `pinneapple-flow/1`; other versions are refused (`FlowError`) |
| `nodes[].id` | a letter or `_`, then letters, digits, `_`, `-` (at most 64); unique; at most 200 nodes |
| `nodes[].type` | a registered node type; nothing else runs: the document never carries code |
| `nodes[].params` | checked against the type's parameter schema (`params_schema()`); unknown names are errors |
| `edges[]` | `source.source_port` (an output) to `target.target_port` (an input) of compatible port types |
| `meta` | free: the planner that produced the flow, the prompt, the stored version |

`pinneapple_flow.validate_flow(doc)` returns every problem (`Issue(level, message, node, edge, param)`): unknown types
or parameters, values out of range, edges between unknown ports or incompatible types, a second edge into a
single-valued input, missing required inputs, cycles. `check_flow` raises when there is an error.
`flow_json_schema()` is the JSON Schema of the document (draft 2020-12).

## Port types

| Type | Value |
|---|---|
| `geometry` | `{"dimension", "shape", "bounds": {"x": [lo, hi], ...}, "time": [t0, t1] or null, "coords": [...]}` |
| `mesh` | `{"geometry", "n": {"x": 64, ...}, "n_t"}` |
| `equations` | `{"fields", "pdes", "conditions", "coefficients", "unknowns", "exact", "family", "classification"}` |
| `problem` | `{"geometry", "equations", "mesh"?}` |
| `solution` | artifact `.npz` with `values` (shape `(*axis lengths, n_fields)`) and `axis_<c>`; the port value carries `coords`, `fields`, `method`, `label` |
| `model` | artifact (`.pt`): a surrogate (`pinneapple_flow.learn.load`) or PINN weights |
| `dataset` | artifact `.npz`: `X, Y` (point observations) or `A, Y, grid, grid_shape` (operator learning) |
| `series` | artifact `.npz`: `t`, `values`; the port value carries `channels` |
| `metrics`, `params` | JSON |
| `figure`, `scene`, `report` | artifacts (PNG; glTF + `pinneapple-field-view/v1` meta; HTML / Markdown / JSON) |
| `twin` | artifact JSON: snapshot, alerts, forecast |
| `any` | accepts every type |

An artifact value is `{"$artifact": "<absolute path in the run folder>", "kind": "npz" | "json" | "pt" | "png" | ...}`;
`pinneapple_flow.runtime.load_artifact(value)` reads it. An output may feed an input of the same type, an input of
type `any`, or one of the conversions in `pinneapple_flow.types.CONVERSIONS` (a `problem` carries its `geometry` and
`equations`, a `solution` is a `dataset`, ...).

## Equations

Equations are text in the grammar of `pinneapple_physics.symbolic_pde.text`: numbers, `+ - * / ** ^`, coordinates
(single letters, `t` is time), fields, coefficient names, `sin cos tan exp log sqrt abs sinh cosh tanh atan asin acos
sign heaviside`, derivative shorthand (`u_t`, `u_xx`, `T_xy`), `d(f, x, ...)` and `lap(f)` (over the spatial
coordinates, so one text works in 1-D, 2-D and 3-D). The text is translated node by node from Python's `ast`; it is
never passed to `eval` or `sympify`. Conditions are `{"kind": "dirichlet" | "neumann" | "robin" | "initial" |
"periodic", "on": "x_min" | "x_max" | ... | "boundary" | "t_min", "equation": "u = 0"}`.

## Adding node types from another package

```python
from pinneapple_flow import Param, Port, node

@node("cfd.simple_foam", "simpleFoam case", category="CFD",
      inputs=[Port("mesh", "mesh")], outputs=[Port("solution", "solution"), Port("metrics", "metrics")],
      params=[Param("iterations", "int", 500, minimum=1)],
      calls="pinneapple_cfd.solvers.simple_foam")
def simple_foam(ctx, inputs, params):
    ...                                   # ctx.workdir, ctx.log, ctx.metric, ctx.figure, ctx.save_npz, ctx.view
    return {"solution": ctx.save_npz("solution", values=..., axis_x=...), "metrics": {...}}
```

Import the module that defines the nodes before running or serving (for example in the app's start-up); the catalog,
the validator, the planner's repair step and the GUI pick the new types up from `REGISTRY`. Port types are fixed by
the library (add a new one in `pinneapple_flow.types` with a pull request) so documents stay portable.

## Running and storing

* `pinneapple_flow.run(flow, workdir=..., from_node=..., store=...)` compiles the flow into a
  `pinneapple_core.workflows.Workflow` (one step of kind `pinneapple_flow.node` per node) and runs it with
  `WorkflowRunner`: per-node logs and states (`queued`, `running`, `done`, `failed`, `cached`, `cancelled`), retries,
  cancel, and skip-done by content hash (a node whose type, parameters and upstream are unchanged reuses its outputs).
* `pinneapple_flow.store.FlowStore(url)` keeps flows with versions, runs and per-node outputs (metrics, report,
  artifact paths) in SQLite (default) or any SQLAlchemy URL (`postgresql+psycopg://...`). Tables: `flows`,
  `flow_versions`, `runs`, `node_results`.
* `pinneapple_flow.service.FlowService` is what an HTTP layer exposes (plan, validate, previews, flows, runs, node
  results, artifacts); Flow Studio's routes are one call each.

## Decision records

Every choice a flow makes carries a record (`pinneapple_flow.decisions.Decision`, JSON):

```json
{
  "id": "3f2a9c1d0e", "node": "eq", "topic": "equations",
  "what": "equations: u_t = alpha*lap(u)",
  "decided_by": {"kind": "rule", "id": "family.heat"},
  "rationale": "keyword scores heat (4.5), ...; highest score wins",
  "value": ["u_t = alpha*lap(u)"],
  "references": [{"key": "carslaw1959heat", "title": "Conduction of Heat in Solids (2nd ed.)",
                  "authors": ["H. S. Carslaw", "J. C. Jaeger"], "year": 1959, "status": "registry", "link": ""}],
  "evidence": [{"kind": "example", "ref": "examples/flow_studio/results/heat_1d/summary.json",
                "detail": "heat 1D: FD and PINN vs exact, Lab tier"}],
  "uncited_reason": "", "confidence": 0.85, "label": "high", "no_reference": false, "suggestion": "",
  "notes": "", "created": 1760817600.0
}
```

| Field | Meaning |
|---|---|
| `topic` | `equations`, `dimension`, `bcs`, `coefficients`, `method`, `architecture`, `hyperparameters`, `hpo`, `twin`, `validation`, `route`, `data`, `other` |
| `decided_by` | `{"kind": "rule", "id": <rule id in pinneapple_flow.rules>}`, `{"kind": "llm", "id": "<provider>/<model>"}` or `{"kind": "user", "id": ...}` |
| `references` | entries of the curated registry (`pinneapple_flow/references.yaml`, `status: "registry"`), or DOIs / URLs given by an LLM or a user: `verified` when they resolved (only with `PINNEAPPLE_FLOW_VERIFY_REFS=1`), `not found`, or `unverified` |
| `evidence` | what in the repository backs the decision: a test, an example result, a library constant |
| `confidence`, `label` | 0-1 and high / medium / low; a decision with neither a usable reference nor evidence is `no_reference: true`, capped at 0.35 (0.5 for a user's own choice) and carries a `suggestion` (run a validation node or a benchmark) |
| `uncited_reason` | why a rule has no literature reference (reading the user's words, a project convention, a rule of thumb) |
| `raise_with` | for a `no reference` or low-confidence decision: registered node types that would back it with evidence (`pinneapple_flow.decisions.RAISE_NODES`, by topic); the GUI inserts them wired into the flow |

Where they live: planner and LLM decisions in `flow.meta["decisions"]`; user edits in `flow.meta["user_edits"]`
(`{"node", "param", "value", "notes"?, "references"?}`, turned into user decisions by
`pinneapple_flow.decisions.collect`); decisions a node takes while running in its report (`report["decisions"]`). The
store keeps all of them per flow version and per run (table `decisions`); `export.report` prints a "Decisions and
references" section. `tests/test_flow_decisions.py` checks that every rule cites the registry or says why not.

## Lab seal and what would raise it

`pinneapple_flow.lab_status` reads, never computes, the quality letter of a validated item:

* `seal_status(item, lab)` gives `{"letter", "name", "scale", "reason", "failing_check", "hint", "sealed", "source"}`.
  The letter comes from the Lab curation record (`pinneapple_lab.curation`, tiers A-D today) or from its seal
  letter (A-F, with the failing check and how to raise it) once the Lab provides that scale. `letter: null` means
  the item was never curated.
* `next_steps(item, lab)` calls `pinneapple_lab.curation.next_steps` when it exists. Each gap keeps the Lab's
  `action` / `command` / `params`, and `nodes` holds the registered node types that carry the action out: the
  action names a node type, or a library call a registered node wraps. Without that API the result is
  `available: false` with the curation's plain gaps, and no node suggestions.
* `insert_nodes(flow, anchor, [{"type", "params"}])` adds registered node types and wires each input by port
  type: from the anchor first, then its upstream nodes, then any other node. One output feeds at most one input of
  a new node.

`validate.lab` stores the first two under `report["lab"]`. The service attaches the run's seals to every node
result (`lab`). HTTP: `GET /api/lab/seal?item=`, `GET /api/lab/next-steps?item=`, `POST /api/flows/insert-nodes`.

## Library-call nodes and user-defined node types

`lib.<key>` nodes come from the curated allowlist in `pinneapple_flow/libcalls_catalog.py`: a callable
(`module:function` or `module:Class.method`), its palette category, references, which inputs come from ports, and
parameters generated from its signature. A user-defined type is data, never code:

```json
{"type": "custom.soft_soil", "base": "solver.terramechanics", "title": "Wheel on JSC-1A", "category": "Custom",
 "params": {"soil": {"default": "JSC-1A", "hidden": true}, "slip": {"default": 0.3}}}
```

`base` must be an allowlisted key and every parameter must exist on it and pass its check. Definitions are stored
(`custom_nodes` table) or carried in `flow.meta["custom_nodes"]` so a flow that uses them stays portable.
`python -m pinneapple_flow coverage` lists which library packages and modules are reached by nodes, and why the
others are not (`docs/flow_studio/COVERAGE.md`).
