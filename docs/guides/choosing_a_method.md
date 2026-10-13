# Choosing a method: PINN, neural operator, graph network or the solver

PINNeAPPle covers the whole Physics-AI workflow: geometry, simulation campaigns, data-driven surrogates (neural
operators, graph networks, MLPs), physics-informed networks, optimisation on top, and verification of the result.
Which learned method to use depends on the problem, not on the library. This page gives the decision table, the
evidence behind it, and the entry point for each choice.

The evidence is an equal-budget benchmark: every method gets the same training compute, the same number of
hyper-parameter trials and the same hardware, and the cost of the simulation data is counted separately
([PINNeAPPle-Benchmark, experiment 09](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/09_equal_budget_pinn_fno_deeponet)).
Its budgets are small (one CPU core per run, no GPU), so read its numbers as directions, not as the best each method
can do.

## Decision table

| Your situation | Use | Why (benchmark evidence below) | Entry point |
|---|---|---|---|
| **No simulation data**, one forward problem with known equations, smooth solution | PINN | Needs only the equations; costs one training per instance | [PINN factory](#pinn-factory) |
| **No simulation data**, coefficients with jumps, shocks, long time horizons, turbulence | The numerical solver | PINNs did not converge on these within the budget; the solver is exact by construction and fast | [Solvers and `pp.cfd`](#solvers-cfd-and-campaigns) |
| **Simulation data available** (or affordable), many new inputs to answer | Neural operator (FNO, DeepONet) | Orders of magnitude more accurate than per-instance PINNs at equal training compute, milliseconds per query | [Neural operators](#neural-operators) |
| **Many geometries**, fields needed on a mesh | Graph network (MeshGraphNet; #293 for Transolver, GINO, point clouds) | Learns on the mesh of every design; gives fields and integrals | [MeshGraphNet](#meshgraphnet-and-geometry-surrogates) |
| **Many geometries**, only integrals needed (Cl, Cd, peak stress) | MLP on the shape parameters | Most accurate and cheapest when the geometry has a low-dimensional parametrisation | [MLP surrogates](#mlp-surrogates) |
| **Inverse problem** (a coefficient from sparse sensors), few cases, parameter possibly outside any simulated range | PINN with the parameter as an inverse unknown, or calibration through the solver | Works from the readings and the equations; does not extrapolate a surrogate | [Inverse problems](#inverse-problems) |
| **Inverse problem**, many cases inside a known range, answers needed in milliseconds | Surrogate + calibration | Pays the data once, then each calibration is a few surrogate calls | [Inverse problems](#inverse-problems) |
| **Fields** versus **integrals** | Fields: operator or graph network. Integrals only: MLP on the parameters | Integral targets are low-dimensional and easier; field models carry more | — |
| **Regime**: laminar or smooth | Any; PINNs are competitive only here | | — |
| **Regime**: turbulent (RANS/LES), shocks, separated flow | Solver for the data, then a data-driven surrogate | A PINN needs a closure it does not have; without data it cannot see the boundary layer or the wake | [Solvers and campaigns](#solvers-cfd-and-campaigns) |

Two costs decide between "train once on data" and "solve each instance":

- **Amortisation.** A surrogate pays for its data and training once, then answers each query in milliseconds. A
  per-instance PINN pays its training for every new instance. `pinneapple_analysis.cost.AmortizedCost` gives the
  cost per query and the break-even number of queries.
- **Extrapolation.** A surrogate is only as good as its training range; a PINN or the solver has no training range.
  Check the out-of-distribution rows below before trusting a surrogate outside its data.

<!-- EVIDENCE -->

## Entry points

### PINN factory

`pinneapple_physics.pinn_solver.factory.pinn_factory`: `PINNProblemSpec` (equations as strings, conditions, inverse
parameters, exogenous signals), `PINNFactory(spec).generate_loss_function()`, `PINN` and `NeuralNetwork`. Hard
initial conditions from gridded data: `pinneapple_simulation.numerical_solvers.operator_benchmarks.FourierInterpolant`.
See [PINN / Physics](../core_concepts/pinn.md).

### Neural operators

`pinneapple_neural.architectures.neural_operators`: `fno.FourierNeuralOperator` (1-D), `fno.FNO2d`, `fno.FNO3d`,
`deeponet.DeepONet`. Reference data for the standard problems: `operator_benchmarks.burgers_1d`,
`navier_stokes_2d_vorticity`, `darcy_2d`, `periodic_grf`.

### MeshGraphNet and geometry surrogates

`pinneapple_neural.architectures.graphnn.mesh_graph_net.MeshGraphNet`; the aero app's field-and-coefficient model
`pinneapple_design.aero.surrogate.AeroGNN`. Geometry-agnostic surrogates (Transolver, GINO, point-cloud baselines
behind one interface): issue #293.

### MLP surrogates

`pinneapple_design.aero.surrogate.MLPEnsemble` (shape parameters + angle → Cl, Cd, Cm, with an ensemble spread);
`pinneapple_neural.architectures.neural_operators.deeponet.mlp` for a plain MLP.

### Inverse problems

`pinneapple_analysis.inverse_problems.heat_transfer`: `run_fin`, `run_plate` (PINN with h as an inverse parameter;
`run_plate(budget=...)` trains under a compute budget). Calibration through a solver or a surrogate:
`pinneapple_simulation.numerical_solvers.heat_fv_reference.plate_2d` plus any 1-D search.

### Solvers, `pp.cfd` and campaigns

`pp.cfd` (`pinneapple_simulation.numerical_solvers.external_flow`): OpenFOAM around any body, `FlowResult`. The
campaign runner `pinneapple_simulation.external_solvers.openfoam.campaign` runs one verified case many times and
returns results a surrogate can train on ([OpenFOAM parameter campaigns](../core_concepts/openfoam_campaigns.md)).
The 2-D airfoil campaign of the benchmark is `apps/aero_optimizer/tools/generate_dataset.py`.

### Comparing methods yourself

`pinneapple_analysis.cost`: `ComputeBudget`, `train_within_budget` (stops at the budget, budget-aware schedules),
`count_flops` (works around `torch.autograd.grad`), `equal_budget_search` (the same number of trials per method),
`AmortizedCost`, `relative_l2`, `time_per_query`, `hardware_record`.

```python
from pinneapple_analysis.cost import ComputeBudget, train_within_budget, cosine_lr, set_lr

budget = ComputeBudget(wall_s=300)            # or flops=..., steps=...

def step(progress):                           # progress in [0, 1]: the fraction of the budget used
    set_lr(opt, cosine_lr(progress, 1e-3))
    opt.zero_grad()
    loss = loss_fn(model, next_batch())
    loss.backward()
    opt.step()
    return loss.detach()

run = train_within_budget(step, budget)       # run.steps, run.wall_s, run.flops, run.history
```
