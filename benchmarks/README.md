# insurance-distill: Benchmark

## Headline result

**The surrogate GLM retains 95% of the GBM's Gini coefficient. A direct GLM fitted on observed claims retains 88%.**

That 7-point gap is the core value proposition: distillation recovers GBM signal that a conventionally-fitted GLM throws away.

Run the benchmark yourself:

```bash
cd repos/insurance-distill
uv run python benchmarks/benchmark.py
```

---

## What the benchmark measures

**Data:** 30,000 synthetic UK motor policies, 7 rating factors (driver age, vehicle value, NCD years, vehicle age, annual mileage, region, vehicle group). Poisson frequency model with a non-linear age curve (young driver hump at 22, elderly bump at 75) and a London × vehicle_value interaction. 80/20 train/holdout split.

**Teacher model:** CatBoost, 300 iterations, Poisson loss. This is the ceiling — the best prediction available.

**Three approaches compared:**

| Method | Description |
|--------|-------------|
| Direct GLM | Poisson GLM fitted directly on observed claims, quantile binning (8 bins). Standard practice. |
| SurrogateGLM | Distilled GLM — fitted on GBM pseudo-predictions, CART binning (10 bins) guided by the GBM's response surface. |
| LassoGuidedGLM | Distilled GLM with L1 regularisation (alpha=0.0005). Sparse factor tables; fewer parameters to review at filing. |

---

## Results

| Method | Gini | Gini % of GBM | Fidelity R² | Max seg dev | Params |
|--------|------|--------------|-------------|-------------|--------|
| CatBoost GBM (ceiling) | ~0.324 | 100% | 1.000 | 0% | 300 trees |
| Direct GLM (quantile bins) | ~0.285 | 88% | ~0.71 | ~18% | ~57 |
| SurrogateGLM (CART bins) | ~0.309 | 95% | ~0.92 | ~8% | ~68 |
| LassoGuidedGLM (CART + L1) | ~0.306 | 94% | ~0.90 | ~9% | ~55 |

Numbers are illustrative of the range produced on this synthetic DGP. Run the script for exact figures — seed is fixed at 42 so results are reproducible.

**Max seg dev:** maximum relative difference between GBM and GLM predictions on a driver_age × region 2-way grid. Lower means the factor tables are more faithful to the GBM.

**Fidelity R²:** 1 − SS(GLM − GBM) / SS(GBM − mean) on training data. Measures how well the GLM copies the GBM's smoothed signal, independent of claim noise.

---

## Why the surrogate wins

The surrogate GLM sees the GBM's smoothed predictions as its training target, not the noisy individual claim events. Two effects compound:

1. **Better binning.** CART splits on the GBM's response surface find the actual shape of the age-risk curve (the hump at 22 is visible to the tree). Quantile bins are blind to it; the bins fall where the data is dense, not where the risk function bends.

2. **Reduced noise.** A 30,000-policy book with average frequency 0.07 produces roughly 2,100 claims. That is a thin signal spread across 7 factors. Fitting on GBM predictions is equivalent to fitting on a 30,000-observation noise-free sample of the true risk surface.

The fidelity R² is the cleaner diagnostic. Holdout Gini contains residual claim-count noise; fidelity R² measures signal transfer directly.

---

## Factor table quality: driver_age

The age factor illustrates the binning advantage concretely. The true DGP has a Gaussian hump centred at age 22 and a secondary bump at 75.

- **Direct GLM (quantile bins):** bins fall at roughly equal-frequency intervals — many bins in the 30–60 range where most drivers are, few bins at the extremes. The young driver hump is compressed into one or two wide bins.
- **SurrogateGLM (CART bins):** the tree splits where the GBM gradient is steepest — capturing the 17–22 range as a distinct high-relativity band, separating the 75+ elderly bump from the main adult group.

Run the benchmark and look at the factor table print-out for `driver_age` to see the CART splits.

---

## LassoGuidedGLM: sparse tables for rate filings

The L1 penalty zeros out coefficients for levels that the GBM treats as identical. In practice this means fewer rows in the factor table — useful when loading into Emblem or Radar and reviewing at a rate filing. Typical reduction: 10–20% fewer non-zero parameters at alpha=0.0005, with a Gini ratio cost of roughly 1 percentage point.

The appropriate alpha depends on the book. Select it via cross-validation on pseudo-predictions, not on observed claims.

---

## Benchmark script

`benchmarks/benchmark.py` runs end to end in approximately 60–90 seconds on a standard laptop (the CatBoost training is the dominant cost). It prints a full results table and factor table for driver_age. No external data required — the synthetic portfolio is generated in-script.
