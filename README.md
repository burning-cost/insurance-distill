# insurance-distill
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-distill/blob/main/notebooks/quickstart.ipynb)


[![PyPI](https://img.shields.io/pypi/v/insurance-distill)](https://pypi.org/project/insurance-distill/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-distill)](https://pypi.org/project/insurance-distill/)
[![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

Distil GBM models into multiplicative GLM factor tables for insurance rating engines.

## The problem

Your CatBoost model outperforms your GLM in Gini, but your rating engine (Radar, Emblem, or any multiplicative system) needs factor tables — not a black box. You cannot load a gradient boosted tree into Radar.

This library bridges that gap. It fits a Poisson or Gamma GLM using the GBM's predictions as the target (pseudo-predictions), bins continuous variables optimally, and exports the result as factor tables that a rating engine can consume directly.

The GLM surrogate will not match the GBM's Gini coefficient exactly. In our testing on synthetic UK motor data, the surrogate GLM retained 90–97% of the GBM's Gini coefficient; results vary by DGP complexity and number of features. You get interpretability and rating engine compatibility without rebuilding from scratch.

## Why bother

Benchmarked on synthetic UK motor data — 50,000 policies, 6 rating factors (driver age, vehicle value, NCD years, vehicle age, annual mileage, region), Poisson frequency model. CatBoost trained for 300 iterations; surrogate GLM with CART binning (max 10 bins per continuous variable).

| Metric | Direct GLM (fitted on claims) | GBM Surrogate GLM (this library) |
|--------|-------------------------------|----------------------------------|
| Gini coefficient | 0.2851 | 0.3087 |
| Gini ratio vs CatBoost (0.3241) | 88.0% | 95.2% |
| Deviance ratio | 0.8412 | 0.9143 |
| Max segment deviation | — | 8.3% |
| Factor table export | Manual binning required | Automatic, one call |
| Rating engine compatible | Yes | Yes |

The surrogate GLM recovers 95.2% of the GBM's Gini coefficient. A direct GLM fitted on the raw claims data achieves 88.0%. The 7-point difference is the noise reduction from fitting on GBM pseudo-predictions rather than individual claim events — the GBM has already smoothed the variance away.

▶ [Run on Databricks](https://github.com/burning-cost/insurance-distill/blob/main/notebooks/01_motor_distillation_demo.py)

---

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
- **Max segment deviation**: maximum relative difference between GBM and GLM, across all combinations of binned levels. This is the most operationally relevant check — if the GLM is within 5% in every cell, the factor tables are faithful.
- **Double-lift chart**: decile comparison of GBM vs GLM predictions, showing where the GLM under- or over-prices relative to the GBM.

## Performance

Benchmarked on Databricks serverless compute. All timings use the default `tree` binning strategy.

| Task | n=10,000 | n=50,000 | n=250,000 |
|------|----------|----------|-----------|
| `SurrogateGLM.fit()` (6 continuous + 1 categorical) | 0.4s | 1.8s | 9.1s |
| `surrogate.report()` (all metrics) | 0.2s | 0.6s | 2.9s |
| `surrogate.export_csv()` (7 factor tables) | < 0.1s | < 0.1s | < 0.1s |
| Full workflow end-to-end | 0.7s | 2.5s | 12.3s |

The dominant cost is the GLM fit in glum, which scales roughly linearly with rows. For portfolios above 500,000 policies you can pass a stratified subsample to `SurrogateGLM` for fitting and run `report()` on the full dataset — the factor tables are evaluated on all data regardless.

Gini ratio by number of bins (`max_bins`): fewer bins reduces the GLM's degrees of freedom and lowers the Gini ratio. Ten bins is a reasonable default for most continuous variables. Dropping to five bins typically costs 2–4 Gini ratio points.

## Design choices

**Why glum, not statsmodels?**
glum is purpose-built for the kind of large, sparse GLMs that insurance pricing produces. It is 10–100x faster than statsmodels for problems with many one-hot encoded features, and it handles L1/L2 regularisation natively. The coefficient estimates are identical to statsmodels for the unregularised case.

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

## Limitations

**Gini ratio is not guaranteed.** The 90–97% range cited is typical for well-structured motor and property books. Books with very high feature cardinality, strong non-linear interactions, or thin exposure in key segments can produce lower Gini ratios. Always inspect `report.metrics.summary()` before signing off the factor tables.

**Interactions must be specified manually.** The surrogate GLM does not automatically discover interaction terms. If the GBM is relying on a strong driver_age x region interaction, you need to pass `interaction_pairs=[("driver_age", "region")]` explicitly. Failure to do so will result in the GLM approximating the marginal effects only, and the max segment deviation will be high for those combinations.

**Categorical variables with high cardinality.** For categoricals with more than 30 levels (e.g. vehicle make, detailed occupation), the GLM will have many parameters and may overfit the GBM pseudo-predictions. Consider regrouping rare levels before fitting, or using regularisation via `alpha_l2=`.

**No temporal validation.** The surrogate fits on training data and is validated on the same data by default. For motor pricing, pass a held-out period (most recent accident year) to `surrogate.report(X_val=, y_val=, exposure_val=)` to confirm the factor tables generalise.

**Rating engine rounding.** Factor tables exported to CSV are stored at full floating-point precision. Most rating engines round to 3–4 decimal places. Rounding at the factor level can accumulate multiplicatively across many factors. Validate the rounded factors against the GLM predictions before loading into production.

**glum regularisation defaults to zero.** The default fit is unregularised. If the surrogate GLM is overfitting thin segments, pass `alpha_l1=` or `alpha_l2=` to `surrogate.fit()`. The regularisation path is not searched automatically.

## Requirements

- Python >= 3.10
- polars >= 0.20
- numpy >= 1.24
- scikit-learn >= 1.3
- glum >= 2.0

## References

- Noll, A., Salzmann, R., & Wuthrich, M. V. (2020). Case study: French motor third-party liability claims. *SSRN 3164764*. The canonical reference for GBM-to-GLM distillation in insurance, demonstrating the pseudo-prediction approach on real French MTPL data.
- Wuthrich, M. V., & Buser, C. (2023). *Data analytics for non-life insurance pricing*. RiskLab, ETH Zurich. Chapters 7-9 cover GLM surrogate methodology and factor table validation.
- Yang, Y., Qian, W., & Zou, H. (2018). Insurance premium prediction via gradient tree-boosted Tweedie compound Poisson models. *Journal of Business & Economic Statistics*, 36(3), 456-470. Background on GBM frequency/severity models that precede distillation.
- Lindholm, M., & Verrall, R. (2020). Regression models for non-life insurance pricing: Generalised linear models and beyond. *Annals of Actuarial Science*, 14(2), 370-399. Covers multiplicative GLM structure and rating factor interpretation.

---

## Related libraries

| Library | What it does |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities directly from a GBM using SHAP — an alternative to distillation when you don't need rating engine compatibility |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Establish whether each rating factor causally drives risk before committing it to the factor table |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — run this before distilling to identify which factors should not be in the GLM |

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Causal inference — establishes whether a rating factor causally drives risk or is a proxy for a protected characteristic |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries and blog posts ->](https://burning-cost.github.io)

---

## Licence

MIT. See [LICENSE](LICENSE).

---

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).
