# Forecasting near-random series without overfitting

Some series are almost random: chaotic systems observed with noise, noise-driven physical systems, financial
returns. A flexible model always finds *something* in them on its training data. This guide gives measured answers
to four questions:

1. Is there anything predictable at all, and how do you know before fitting a model?
2. How far ahead can each model family forecast before its skill falls back to a baseline?
3. Which training choices actually prevent overfitting, and by how much?
4. How do the point and probabilistic scores compare once the differences are tested for significance?

Everything here comes from one script,
[`examples/near_random_forecasting/run_study.py`](../../examples/near_random_forecasting/run_study.py). Its results
are committed in `examples/near_random_forecasting/results/`: one JSON per series, the overfitting studies, and
`tables.md`, from which every table below is copied. The figures are in `examples/near_random_forecasting/figures/`.
The library pieces live in `pinneapple_systems.time_series` (`predictability`, `synthetic`, `forecast_lab`,
`metrics_ext`, `validation`, `datasets.public`) and are tested in `tests/test_near_random_forecasting.py`.

## Quick start

```python
from pinneapple_systems.time_series.predictability import predictability_report
from pinneapple_systems.time_series.forecast_lab import presets
from pinneapple_systems.time_series.forecast_lab.study import run_zoo

c = presets.case("double_well", n=1500)                       # series, protocol and model zoo
report = predictability_report(c.data.y, horizons=c.protocol.horizons, dt=c.data.dt, n_surrogates=19)
print(report["surrogate_tests"]["npe_iaaft"]["p_value"])      # more structure than a linear Gaussian process?
res = run_zoo(c.data, c.specs[:6], c.protocol, budget=2)      # baselines + gradient boosting, purged CV tuning
print(res["models"]["gbm"]["test"]["skill_rmse"], res["models"]["gbm"]["skill_horizon_vs_best_baseline"])
```

Your own series: wrap it in `SeriesData(obs, target=0, dt=...)`, pick horizons for a `Protocol`, and list
`ModelSpec(name, family, factory, grid)` entries. Every forecaster implements
`fit(data, train_origins, val_origins, horizons)` and `predict(data, origins) -> Forecast` (a mean plus a Gaussian
standard deviation or ensemble samples).

<!-- RESULTS -->
