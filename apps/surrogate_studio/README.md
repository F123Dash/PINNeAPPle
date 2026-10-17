# Surrogate Studio (day 14)

**Live:** https://surrogate.pinneapple.org · **Port:** 8097 · **Prefix:** `SUR`

Turn the simulation results you already have into a fast predictive model, measured on runs it never saw, served
behind a versioned endpoint with its validity range enforced.

## Pain

Every new evaluation needs another run of an expensive solver (CFD, FEA, a thermal model). A surrogate answers in
microseconds, but only if someone picks the right method, tests it honestly on data it did not train on, and says
where it stops being valid. Most teams skip the last two.

## Input

- A table of runs: CSV, Parquet, JSON, HDF5 or NPZ, one row per run. Inputs are the parameters you set; outputs are
  scalars (`Cl`, `Rth`, `pressure_drop`) or a field stored as columns `u_0, u_1, ...` with its grid shape (and
  optionally coordinate names and ranges).
- A `DatasetSpec` (`pinneapple_data.contracts`, embedded in Parquet / HDF5 or as a `.contract.json` sidecar) sets the
  column roles, so the output of the Campaign Planner or the Workflow Runner goes straight in.
- Optional: a group column (geometry or case id, held out whole), a declared PDE (enables the PINN), constraints
  (`Cd > 0`), the budget per method, seeds, the endpoint name and the out-of-distribution policy.
- Or **Try with an example**, offline in about 10 s: the OpenFOAM airfoil campaign of the zoo (scalars, whole airfoils
  held out) or exact viscous Burgers fields (fields + PDE, the steepest fronts held out).

## Output

- An equal-budget comparison of every applicable method on the same splits: validation, held-out test and
  distribution-shift error, spread over seeds, training time, inference time per query, parameter count.
- The chosen model (by validation error; the test and shift sets never choose), hosted at once at
  `/v1/models/<name>/predict` with the Model Gateway's API: versions, OOD gate per query (warn or refuse), run log,
  latency / count / error metrics.
- A limitations report: validity bounds, the OOD gate and how the errors of flagged runs compare, the error per input
  quartile (where it concentrates), the worst runs, the extrapolation ratio, and for fields the error map and the
  PDE residual of the predictions.
- A trust decision (APPROVED / REVIEW / REJECT, `pinneapple_analysis.trust.TrustReport`) with every check and its
  value; a REJECT model is hosted in the `development` stage only.
- Downloads: `study_result.json`, `model_ref.json`, `dataset_spec.json` (the shared contracts), `report.md`,
  `study.json`, `model.zip` and, for PyTorch models, `model.onnx` and `model.pt` (TorchScript).

## Integrations

scikit-learn (ridge, Gaussian process, gradient boosting), XGBoost (optional), PyTorch (MLP, FNO, DeepONet, PINN),
ONNX and TorchScript export, CSV / Parquet / HDF5 / NPZ, the shared contracts (DatasetSpec in, ModelRef and
StudyResult out), REST and Python. JAX is not used.

## Methods (PINNeAPPle)

`pinneapple_analysis.surrogate.run_study`: candidate selection with a reason for each method that does not apply
(fields vs scalars; FNO and DeepONet for fields; a PINN only with a declared PDE; GP up to 2000 samples), equal compute
per method (`pinneapple_analysis.cost.equal_budget`), held-out and distribution-shift splits (`make_splits`),
limitations and OOD gate (`limits`), trust report; `pinneapple_registry.ModelGateway` and `gateway_api` for hosting;
`pinneapple_analysis.surrogate.contracts` for the contracts.

## API

| Route | What |
|---|---|
| `POST /api/uploads` | upload a table (shared runtime) |
| `POST /api/inspect` | columns, fields and suggested roles of a table |
| `POST /api/studies` | start a study (`{"example": ...}` or `{"upload_id", "inputs", "outputs", ...}`); a job |
| `GET /api/jobs/{id}`, `/logs` | progress and result (shared runtime) |
| `GET /api/studies/{id}/files/{name}` | downloads |
| `GET /v1/models`, `POST /v1/models/{name}/predict`, `/runs`, `/v1/metrics`, `.../export/{fmt}` | the hosted models |

Interactive documentation at `/docs`.

## Monetization

Paid training (budget and seeds beyond the free tier, larger tables), metered inferences on hosted models, private
model hosting with API keys, and enterprise deployments next to the customer's solver data.

## Limits

- Errors are measured on held-out runs of the same campaign; another solver setting, mesh or physics regime is not
  covered. The OOD gate looks at the inputs only.
- Each method runs with its library defaults (no per-method hyper-parameter search), so a tuned method may do better.
- The physics checks are the declared PDE residual (finite differences on the grid) and declared constraints;
  conservation and boundary conditions of outputs are not checked here (use `pinneapple check`).
- FNO here takes parameters as constant input channels on 1-D or 2-D uniform grids; irregular meshes are not
  supported. Gaussian processes are limited to 2000 training samples.
- Hosted models are visible to everyone who can reach the server; use the login or `SUR_REQUIRE_API_KEY=1`.
- Thread jobs are cancelled at their next progress report; `SUR_JOB_EXECUTOR=process` kills them at once.

## Run locally

```bash
pip install -e ".[surrogate]" -r apps/surrogate_studio/requirements.txt
cd apps/surrogate_studio && uvicorn surrogate_studio.api:app --port 8097
```
