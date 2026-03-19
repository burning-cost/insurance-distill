# insurance-distill

Distil GBM models into multiplicative GLM factor tables for insurance rating engines.

## The problem

Your CatBoost model outperforms your GLM in Gini, but your rating engine (Radar, Emblem, or any multiplicative system) needs factor tables - not a black box. You cannot load a gradient boosted tree into Radar.

This library bridges that gap. It fits a Poisson or Gamma GLM using the GBM's predictions as the target (pseudo-predictions), bins continuous variables optimally, and exports the result as factor tables that a rating engine can consume directly.

The GLM surrogate will not match the GBM's Gini coefficient exactly. A well-tuned distillation does not match the GBM's Gini coefficient exactly. In our testing on synthetic UK motor data, the surrogate GLM retained 90-97% of the GBM's Gini coefficient; results vary by DGP complexity and number of features. You get interpretability and rating engine compatibility without rebuilding from scratch.

## Installation

```bash
uv add insurance-distill
```

With CatBoost support:

```bash
uv add "insurance-distill[catboost]"
```

## Quick start

```python
from insurance_distill import SurrogateGLM

# fitted_catboost: any sklearn-compatible model (CatBoost, sklearn GBM, etc.)
surrogate = SurrogateGLM(
    model=fitted_catboost,
    X_train=X_train,          # Polars DataFrame
    y_train=y_train,          # actual claim counts or amounts
    exposure=exposure_arr,    # earned car-years (or None for unit exposure)
    family="poisson",         # or "gamma" for severity
)

surrogate.fit(
    max_bins=10,                                   # bins per continuous variable
    interaction_pairs=[("driver_age", "region")],  # optional interaction terms
)

# Validation
report = surrogate.report()
print(report.metrics.summary())
# Gini (GBM):              0.3241
# Gini (GLM surrogate):    0.3087
# Gini ratio:              95.2%
# Deviance ratio:          0.9143
# Max segment deviation:   8.3%
# Mean segment deviation:  2.1%
# Segments evaluated:      312

# Inspect a single factor table
driver_age_table = surrogate.factor_table("driver_age")
print(driver_age_table)
# shape: (8, 3)
# | level              | log_coefficient | relativity |
# | [-inf, 21.00)      | 0.412           | 1.510      |
# | [21.00, 25.00)     | 0.218           | 1.244      |
# ...

# Export all factor tables as CSV (one file per variable)
surrogate.export_csv("output/factors/", prefix="motor_freq_")
# Writes: motor_freq_driver_age.csv, motor_freq_vehicle_value.csv, ...
```

## Binning strategies

Three binning methods are available. The default (`tree`) is the right choice for most variables.

| Method | Description | When to use |
|--------|-------------|-------------|
| `tree` | CART decision tree on GBM pseudo-predictions | Default. Finds statistically meaningful cut-points. |
| `quantile` | Equal-frequency bins | Fallback when the tree produces degenerate splits. |
| `isotonic` | Change-points from isotonic regression | Monotone variables (e.g. no-claims discount, years held). |

You can mix methods per variable:

```python
surrogate.fit(
    max_bins=10,
    binning_method="tree",
    method_overrides={
        "ncd_years": "isotonic",
        "vehicle_age": "quantile",
    },
)
```

## Validation metrics

After fitting, `surrogate.report()` returns a `DistillationReport` with:

- **Gini ratio**: how much of the GBM's discrimination the GLM retains. Above 0.90 is generally acceptable; above 0.95 is excellent.
- **Deviance ratio**: analogous to R-squared for GLMs. Measures how well the GLM explains the GBM's predictions.
- **Max segment deviation**: maximum relative difference between GBM and GLM, across all combinations of binned levels. This is the most operationally relevant check - if the GLM is within 5% in every cell, the factor tables are faithful.
- **Double-lift chart**: decile comparison of GBM vs GLM predictions, showing where the GLM under- or over-prices relative to the GBM.

## Design choices

**Why glum, not statsmodels?**
glum is purpose-built for the kind of large, sparse GLMs that insurance pricing produces. It is 10-100x faster than statsmodels for problems with many one-hot encoded features, and it handles L1/L2 regularisation natively. The coefficient estimates are identical to statsmodels for the unregularised case.

**Why Polars?**
We use Polars for data handling because it is faster and more memory-efficient than pandas for the aggregation operations (segment deviation, lift charts) that this library relies on. The GLM fitting itself uses numpy arrays internally, as glum requires.

**Why pseudo-predictions, not actual claims?**
Fitting the GLM on GBM predictions rather than actual claims eliminates the noise from individual claim events. The GBM has already smoothed over that noise. Fitting the surrogate on the GBM's output gives a cleaner signal for the GLM to learn from, resulting in better-preserved Gini.

**Multiplicative by construction**
The GLM always uses a log link function. This means the factor tables are multiplicative: the final premium is the product of the base rate and each factor. This is the convention used by Radar, Emblem, Guidewire, and most other UK personal lines rating engines.

## Factor table format

Each factor table is a Polars DataFrame with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `level` | str | Bin label (e.g. `[25.00, 40.00)`) or category value |
| `log_coefficient` | float | Raw GLM coefficient on log scale (0.0 for base level) |
| `relativity` | float | Multiplicative factor = exp(log_coefficient) |

The base level (reference category) always has `relativity = 1.0`. All other levels are expressed relative to it.

## Requirements

- Python >= 3.10
- polars >= 0.20
- numpy >= 1.24
- scikit-learn >= 1.3
- glum >= 2.0
