"""
Benchmark: GBM distillation vs direct GLM — how much signal do you keep?

The scenario: a UK motor frequency model. The team has a CatBoost model that
outperforms their current GLM, but the rating system (Emblem/Radar) needs
factor tables. The question is whether distillation recovers more of the
GBM's discriminatory power than a GLM fitted directly on observed claims.

Three approaches are compared on the same holdout set:

  1. Direct GLM  — GLM fitted directly on observed claims with quantile
     binning. This is what most teams do today. It sees only the noisy
     claim events, not the smoothed GBM predictions.

  2. SurrogateGLM  — standard distillation. GLM fitted on GBM pseudo-
     predictions (unregularised, CART binning). The GBM has already
     smoothed individual claim noise away; the surrogate captures that
     smoothed signal.

  3. LassoGuidedGLM  — distillation with L1 regularisation, mimicking
     the Lindholm & Palmquist (2024) approach. Sparse coefficient vector
     means fewer effective parameters; the factor tables are simpler to
     review in a rate filing and easier to load into Radar/Emblem.

     NOTE: LassoGuidedGLM is not yet a separate class in insurance-distill.
     This benchmark implements it as SurrogateGLM with alpha > 0 (L1 penalty
     via glum). A dedicated class with auto-calibration and PD-guided binning
     is on the roadmap.

Synthetic data: 30,000 UK motor policies, 7 rating factors, Poisson frequency
model with a non-linear age curve and a London x vehicle_value interaction.

Metrics reported:
  - Gini coefficient on holdout (discriminatory power, exposure-weighted)
  - Fidelity R² = 1 - SS(GLM - GBM) / SS(GBM - mean) on training data
    This is the most direct measure of how faithfully the GLM reproduces the
    GBM's predictions — independent of claim noise.
  - Deviance ratio on training pseudo-predictions (higher is better)
  - MAE vs true claim rate on holdout
  - Parameter count (factor table rows)
  - Segment deviation on a 2-factor grid (driver_age x region only) to avoid
    sparsity artefacts. With all 7 features the cross-product has ~20k cells
    from 24k rows, making per-cell means unreliable.

Run:
    cd /home/ralph/repos/insurance-distill
    uv run python benchmarks/benchmark.py
"""
from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: GBM distillation vs direct GLM (insurance-distill)")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from insurance_distill import SurrogateGLM, compute_gini, compute_deviance_ratio
    from insurance_distill._binning import OptimalBinner
    print("insurance-distill imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-distill: {e}")
    print("Install with: pip install insurance-distill")
    sys.exit(1)

try:
    from catboost import CatBoostRegressor, Pool
    print("CatBoost available")
except ImportError:
    print("CatBoost not available - install with: pip install catboost")
    sys.exit(1)

try:
    from glum import GeneralizedLinearRegressor
    print("glum available")
except ImportError:
    print("glum not available - install with: pip install glum")
    sys.exit(1)

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict_surrogate_on_holdout(
    surrogate: SurrogateGLM,
    df_test: pl.DataFrame,
) -> np.ndarray:
    """
    Apply the fitted surrogate's binning + GLM to a held-out DataFrame.

    SurrogateGLM stores all binning state after .fit(); we reuse it here to
    produce out-of-sample predictions. This bridges the gap until a formal
    .predict() method lands on the class.
    """
    binner = OptimalBinner()
    X_binned = binner.transform(df_test, surrogate._bin_specs)

    bin_cols = [
        f"{f}__bin"
        for f in surrogate._features
        if f"{f}__bin" in X_binned.columns
    ]
    cat_cols = [c for c in surrogate._categorical_features if c in X_binned.columns]

    col_blocks = []
    for col in bin_cols + cat_cols:
        series = X_binned[col].cast(pl.String)
        relevant = [c for c in surrogate._design_col_names if c.startswith(f"{col}=")]
        for col_name in relevant:
            level = col_name.split("=", 1)[1]
            col_blocks.append((series == level).cast(pl.Int8).to_numpy())

    if not col_blocks:
        raise RuntimeError("No matching columns found for holdout prediction.")

    X_design = np.column_stack(col_blocks).astype(np.float64)
    return np.asarray(surrogate._glm.predict(X_design), dtype=float)


def fit_direct_glm(
    df_train: pl.DataFrame,
    y_train: np.ndarray,
    exposure_train: np.ndarray,
    df_test: pl.DataFrame,
    cont_features: list[str],
    cat_features: list[str],
    max_bins: int,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Fit a Poisson GLM directly on observed claim rates with quantile binning.

    This is the standard "hand-fitted" approach. Binning is not GBM-guided;
    quantile bands are used (equal-frequency, no information about the shape
    of the risk response).

    Returns train_preds, test_preds, n_params.
    """
    col_blocks_train = []
    col_blocks_test = []
    n_params = 1  # intercept

    for feat in cont_features:
        train_vals = df_train[feat].to_numpy().astype(float)
        test_vals = df_test[feat].to_numpy().astype(float)

        quantiles = np.linspace(0, 100, max_bins + 1)
        edges = np.unique(np.percentile(train_vals, quantiles))

        bins_train = np.digitize(train_vals, edges[1:-1])
        bins_test = np.digitize(test_vals, edges[1:-1])

        for lv in range(1, len(np.unique(bins_train))):
            col_blocks_train.append((bins_train == lv).astype(float))
            col_blocks_test.append((bins_test == lv).astype(float))
            n_params += 1

    for feat in cat_features:
        train_cats = df_train[feat].to_numpy().astype(str)
        test_cats = df_test[feat].to_numpy().astype(str)
        levels = sorted(np.unique(train_cats))
        for lv in levels[1:]:
            col_blocks_train.append((train_cats == lv).astype(float))
            col_blocks_test.append((test_cats == lv).astype(float))
            n_params += 1

    X_train_d = np.column_stack(col_blocks_train).astype(np.float64)
    X_test_d = np.column_stack(col_blocks_test).astype(np.float64)

    glm = GeneralizedLinearRegressor(
        family="poisson",
        link="log",
        alpha=alpha,
        fit_intercept=True,
        max_iter=200,
    )
    rate_train = y_train / np.clip(exposure_train, 1e-8, None)
    glm.fit(X_train_d, rate_train, sample_weight=exposure_train)

    return (
        np.asarray(glm.predict(X_train_d), dtype=float),
        np.asarray(glm.predict(X_test_d), dtype=float),
        n_params,
    )


def segment_deviation_2way(
    df: pl.DataFrame,
    pseudo: np.ndarray,
    glm_pred: np.ndarray,
    exposure: np.ndarray,
    feat_a: str,
    feat_b: str,
    n_bins_a: int = 5,
    n_bins_b: int = 5,
) -> tuple[float, float, int]:
    """
    Compute segment deviation on a 2-way grid to avoid sparsity artefacts.

    With 7 features x 10 bins each the full cross-product produces ~20k
    unique cells from 24k rows — most cells have 1 observation, making
    per-cell averages meaningless. Restricting to a 2-way grid (e.g.
    driver_age x region) gives ~40 cells with ~600 rows each.

    Returns max_dev, mean_dev (exposure-weighted), n_segments.
    """
    vals_a = df[feat_a].to_numpy().astype(float)
    edges_a = np.unique(np.percentile(vals_a, np.linspace(0, 100, n_bins_a + 1)))
    bins_a = np.digitize(vals_a, edges_a[1:-1])

    cats_b = df[feat_b].to_numpy().astype(str)
    unique_b = sorted(np.unique(cats_b))

    rows = []
    for bin_a in np.unique(bins_a):
        for cat_b in unique_b:
            mask = (bins_a == bin_a) & (cats_b == cat_b)
            if not mask.any():
                continue
            w = exposure[mask]
            avg_pseudo = float((pseudo[mask] * w).sum() / w.sum())
            avg_glm = float((glm_pred[mask] * w).sum() / w.sum())
            rel_dev = abs(avg_glm - avg_pseudo) / max(avg_pseudo, 1e-10)
            rows.append({"rel_dev": rel_dev, "exp": float(w.sum())})

    if not rows:
        return 0.0, 0.0, 0

    devs = np.array([r["rel_dev"] for r in rows])
    exps = np.array([r["exp"] for r in rows])
    max_dev = float(devs.max())
    mean_dev = float((devs * exps).sum() / exps.sum())
    return max_dev, mean_dev, len(rows)


# ---------------------------------------------------------------------------
# 1. Generate synthetic motor portfolio
# ---------------------------------------------------------------------------

N_TOTAL = 30_000
SEED = 42
rng = np.random.default_rng(SEED)

print(f"\n[1] Generating {N_TOTAL:,} synthetic motor policies...")

REGIONS = [
    "London", "South_East", "Midlands", "North_West",
    "North_East", "South_West", "Scotland", "Wales",
]
VEHICLE_GROUPS = ["A", "B", "C", "D", "E"]

region = rng.choice(REGIONS, N_TOTAL)
vehicle_group = rng.choice(VEHICLE_GROUPS, N_TOTAL, p=[0.28, 0.30, 0.22, 0.14, 0.06])
driver_age = np.clip(rng.normal(42, 14, N_TOTAL), 17, 85).astype(int)
vehicle_value = np.clip(rng.lognormal(9.5, 0.6, N_TOTAL), 2_000, 80_000)
ncd_years = rng.integers(0, 10, N_TOTAL)
vehicle_age = rng.integers(0, 20, N_TOTAL)
annual_mileage = np.clip(rng.lognormal(9.6, 0.5, N_TOTAL), 1_000, 60_000)
exposure = np.clip(rng.beta(8, 2, N_TOTAL), 0.1, 1.0)

region_loading = {
    "London": 0.35, "South_East": 0.20, "Midlands": 0.08,
    "North_West": 0.12, "North_East": 0.10, "South_West": -0.05,
    "Scotland": -0.10, "Wales": -0.08,
}
vg_loading = {"A": -0.10, "B": 0.00, "C": 0.15, "D": 0.32, "E": 0.55}

reg_arr = np.array([region_loading[r] for r in region])
vg_arr = np.array([vg_loading[v] for v in vehicle_group])

# Non-linear age: young driver hump (peak ~22) + elderly bump (peak ~75)
age_effect = (
    0.45 * np.exp(-0.5 * ((driver_age - 22) / 4.0) ** 2)
    + 0.18 * np.exp(-0.5 * ((driver_age - 75) / 6.0) ** 2)
    - 0.018 * np.clip(driver_age - 30, 0, 30)
)

# London x vehicle_value interaction (theft risk)
london_mask = (region == "London").astype(float)
value_london = 0.12 * london_mask * (np.log(vehicle_value / 15_000))

log_rate_true = (
    -2.8
    + age_effect
    + reg_arr
    + vg_arr
    + 0.18 * np.log(vehicle_value / 15_000)
    - 0.06 * np.minimum(ncd_years, 7)
    + 0.015 * vehicle_age
    + 0.10 * np.log(annual_mileage / 10_000)
    + value_london
)

true_rate = np.exp(log_rate_true)
claims = rng.poisson(true_rate * exposure).astype(float)

df = pl.DataFrame({
    "driver_age": driver_age,
    "vehicle_value": vehicle_value,
    "ncd_years": ncd_years.astype(np.int32),
    "vehicle_age": vehicle_age.astype(np.int32),
    "annual_mileage": annual_mileage,
    "region": region,
    "vehicle_group": vehicle_group,
})

print(f"   Policies:       {N_TOTAL:,}")
print(f"   Total claims:   {int(claims.sum()):,}")
print(f"   Avg frequency:  {claims.sum() / exposure.sum():.4f}")

n_train = int(0.8 * N_TOTAL)
idx_all = np.arange(N_TOTAL)
rng.shuffle(idx_all)
train_idx = idx_all[:n_train]
test_idx = idx_all[n_train:]

df_train = df[train_idx]
df_test = df[test_idx]
y_train = claims[train_idx]
exp_train = exposure[train_idx]
exp_test = exposure[test_idx]
true_rate_test = true_rate[test_idx]

print(f"   Train: {n_train:,}  |  Holdout: {len(test_idx):,}")

CONT_FEATURES = ["driver_age", "vehicle_value", "ncd_years", "vehicle_age", "annual_mileage"]
CAT_FEATURES = ["region", "vehicle_group"]

# ---------------------------------------------------------------------------
# 2. CatBoost GBM (the "teacher")
# ---------------------------------------------------------------------------

print(f"\n[2] Training CatBoost frequency model (teacher)...")
t0 = time.time()

X_train_pd = df_train.to_pandas()
X_test_pd = df_test.to_pandas()

gbm = CatBoostRegressor(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Poisson",
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False,
)
train_pool = Pool(
    X_train_pd,
    label=y_train / exp_train,
    cat_features=CAT_FEATURES,
    weight=exp_train,
)
gbm.fit(train_pool)
gbm_train_time = time.time() - t0

gbm_preds_train = gbm.predict(X_train_pd)
gbm_preds_test = gbm.predict(X_test_pd)

gini_gbm = compute_gini(gbm_preds_test, exp_test)
mae_gbm = float(np.average(np.abs(gbm_preds_test - true_rate_test), weights=exp_test))

# Denominator for fidelity R²
ss_tot = float(np.sum((gbm_preds_train - gbm_preds_train.mean()) ** 2))

print(f"   Train time:     {gbm_train_time:.1f}s")
print(f"   Gini (holdout): {gini_gbm:.4f}")
print(f"   MAE vs truth:   {mae_gbm:.5f}")

# ---------------------------------------------------------------------------
# 3. Direct GLM — fitted on observed claims, quantile binning
# ---------------------------------------------------------------------------

print(f"\n[3] Fitting direct GLM (quantile binning, observed claims)...")
t0 = time.time()

direct_train_preds, direct_test_preds, n_params_direct = fit_direct_glm(
    df_train, y_train, exp_train, df_test,
    cont_features=CONT_FEATURES,
    cat_features=CAT_FEATURES,
    max_bins=8,
)
direct_time = time.time() - t0

gini_direct = compute_gini(direct_test_preds, exp_test)
mae_direct = float(np.average(np.abs(direct_test_preds - true_rate_test), weights=exp_test))
ss_res_direct = float(np.sum((direct_train_preds - gbm_preds_train) ** 2))
r2_direct = 1.0 - ss_res_direct / max(ss_tot, 1e-10)
dev_ratio_direct = compute_deviance_ratio(gbm_preds_train, direct_train_preds, "poisson")

max_seg_direct, mean_seg_direct, n_seg_direct = segment_deviation_2way(
    df_train, gbm_preds_train, direct_train_preds, exp_train,
    feat_a="driver_age", feat_b="region",
)

print(f"   Fit time:       {direct_time:.1f}s")
print(f"   Gini (holdout): {gini_direct:.4f}  (GBM: {gini_gbm:.4f})")
print(f"   Gini ratio:     {gini_direct/gini_gbm:.1%}")
print(f"   R² vs GBM:      {r2_direct:.4f}")
print(f"   Dev ratio:      {dev_ratio_direct:.4f}")
print(f"   Max seg dev:    {max_seg_direct:.1%}  (driver_age x region grid, {n_seg_direct} cells)")
print(f"   Parameters:     {n_params_direct}")

# ---------------------------------------------------------------------------
# 4. SurrogateGLM — CART binning on GBM pseudo-predictions, unregularised
# ---------------------------------------------------------------------------

print(f"\n[4] Distilling GBM -> SurrogateGLM (CART binning, unregularised)...")
t0 = time.time()

surrogate = SurrogateGLM(
    model=gbm,
    X_train=df_train,
    y_train=y_train,
    exposure=exp_train,
    family="poisson",
)
surrogate.fit(
    features=CONT_FEATURES,
    categorical_features=CAT_FEATURES,
    max_bins=10,
    binning_method="tree",
    method_overrides={"ncd_years": "isotonic"},  # monotone variable
)
surrogate_report = surrogate.report()
surrogate_time = time.time() - t0

surrogate_preds_test = predict_surrogate_on_holdout(surrogate, df_test)
gini_surrogate = compute_gini(surrogate_preds_test, exp_test)
mae_surrogate = float(np.average(np.abs(surrogate_preds_test - true_rate_test), weights=exp_test))
ss_res_surrogate = float(np.sum((surrogate._glm_predictions - gbm_preds_train) ** 2))
r2_surrogate = 1.0 - ss_res_surrogate / max(ss_tot, 1e-10)
n_params_surrogate = len(surrogate._design_col_names) + 1

max_seg_surrogate, mean_seg_surrogate, n_seg_surrogate = segment_deviation_2way(
    df_train, gbm_preds_train, surrogate._glm_predictions, exp_train,
    feat_a="driver_age", feat_b="region",
)

print(f"   Fit time:       {surrogate_time:.1f}s")
print(f"   Gini (holdout): {gini_surrogate:.4f}  (GBM: {gini_gbm:.4f})")
print(f"   Gini ratio:     {gini_surrogate/gini_gbm:.1%}")
print(f"   R² vs GBM:      {r2_surrogate:.4f}")
print(f"   Dev ratio:      {surrogate_report.metrics.deviance_ratio:.4f}")
print(f"   Max seg dev:    {max_seg_surrogate:.1%}  ({n_seg_surrogate} cells)")
print(f"   Parameters:     {n_params_surrogate}")

# ---------------------------------------------------------------------------
# 5. LassoGuidedGLM — CART binning + L1 regularisation
# ---------------------------------------------------------------------------

print(f"\n[5] Distilling GBM -> LassoGuidedGLM (CART + L1)...")
print(f"    (Sparse distillation after Lindholm & Palmquist 2024)")
t0 = time.time()

# alpha=0.0005: mild sparsity preserving discrimination while reducing parameter count.
# In production, select alpha via cross-validation on pseudo-predictions.
LASSO_ALPHA = 0.0005

lasso_surrogate = SurrogateGLM(
    model=gbm,
    X_train=df_train,
    y_train=y_train,
    exposure=exp_train,
    family="poisson",
    alpha=LASSO_ALPHA,
)
lasso_surrogate.fit(
    features=CONT_FEATURES,
    categorical_features=CAT_FEATURES,
    max_bins=10,
    binning_method="tree",
    method_overrides={"ncd_years": "isotonic"},
)
lasso_report = lasso_surrogate.report()
lasso_time = time.time() - t0

lasso_preds_test = predict_surrogate_on_holdout(lasso_surrogate, df_test)
gini_lasso = compute_gini(lasso_preds_test, exp_test)
mae_lasso = float(np.average(np.abs(lasso_preds_test - true_rate_test), weights=exp_test))
ss_res_lasso = float(np.sum((lasso_surrogate._glm_predictions - gbm_preds_train) ** 2))
r2_lasso = 1.0 - ss_res_lasso / max(ss_tot, 1e-10)
n_params_lasso = len(lasso_surrogate._design_col_names) + 1

coeff_df = lasso_report.glm_coefficients
n_zero_coeffs = 0
if coeff_df is not None and "coefficient" in coeff_df.columns:
    n_zero_coeffs = int((coeff_df["coefficient"].abs() < 1e-6).sum())

max_seg_lasso, mean_seg_lasso, n_seg_lasso = segment_deviation_2way(
    df_train, gbm_preds_train, lasso_surrogate._glm_predictions, exp_train,
    feat_a="driver_age", feat_b="region",
)

print(f"   Fit time:       {lasso_time:.1f}s")
print(f"   L1 alpha:       {LASSO_ALPHA}")
print(f"   Gini (holdout): {gini_lasso:.4f}  (GBM: {gini_gbm:.4f})")
print(f"   Gini ratio:     {gini_lasso/gini_gbm:.1%}")
print(f"   R² vs GBM:      {r2_lasso:.4f}")
print(f"   Dev ratio:      {lasso_report.metrics.deviance_ratio:.4f}")
print(f"   Max seg dev:    {max_seg_lasso:.1%}  ({n_seg_lasso} cells)")
print(f"   Params total:   {n_params_lasso}  ({n_zero_coeffs} lasso-zeroed)")

# ---------------------------------------------------------------------------
# 6. Factor table: driver_age
# ---------------------------------------------------------------------------

print(f"\n[6] Factor table: driver_age")
print("-" * 60)
print(f"  True DGP: non-linear hump at age 22, elderly bump at 75.")
print(f"  Direct GLM uses quantile bins (blind to the GBM's age curve).")
print(f"  SurrogateGLM uses CART splits that respect the GBM's response shape.")
print()

print(f"  SurrogateGLM (CART binning on GBM predictions):")
try:
    age_table = surrogate.factor_table("driver_age")
    level_col = "level" if "level" in age_table.columns else age_table.columns[0]
    for row in age_table.iter_rows(named=True):
        print(f"    {row[level_col]:<22}  relativity={row['relativity']:.4f}")
except Exception as e:
    print(f"    [Could not retrieve: {e}]")

# ---------------------------------------------------------------------------
# 7. Double-lift chart (SurrogateGLM)
# ---------------------------------------------------------------------------

print(f"\n[7] Double-lift chart: GBM vs SurrogateGLM (10 deciles, training data)")
print("-" * 60)
print(f"  Rows sorted by GBM/GLM ratio. Decile 1 = where GLM over-predicts most.")
print(f"  Decile 10 = where GLM under-predicts most. Ideal: all ratios near 1.0.")
print()
lift = surrogate_report.lift_chart
if lift is not None:
    print(f"  {'Decile':>7}  {'Avg GBM':>12}  {'Avg GLM':>12}  {'GLM/GBM':>8}")
    print(f"  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*8}")
    for row in lift.iter_rows(named=True):
        ratio_glm_gbm = 1.0 / max(row["ratio_gbm_to_glm"], 1e-10)
        print(
            f"  {row['decile']:>7}   "
            f"{row['avg_gbm']:>10.5f}    "
            f"{row['avg_glm']:>10.5f}   "
            f"{ratio_glm_gbm:>7.3f}"
        )

# ---------------------------------------------------------------------------
# 8. Results summary table
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")
print()
print(f"  {'Method':<28}  {'Gini':>7}  {'Gini%':>7}  {'R²GBM':>7}  {'DevRatio':>9}  {'MaxSeg':>8}  {'Params':>7}")
print(f"  {'-'*28}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*7}")


def _row(name, gini, r2, dev_ratio, max_seg, n_p):
    return (
        f"  {name:<28}  {gini:>7.4f}  {gini/gini_gbm:>6.1%}  "
        f"{r2:>7.4f}  {dev_ratio:>9.4f}  {max_seg:>7.1%}  {n_p:>7}"
    )


print(_row("CatBoost GBM (ceiling)", gini_gbm, 1.0, 1.0, 0.0, int(gbm.get_param("iterations"))))
print(_row("Direct GLM (quantile)", gini_direct, r2_direct, dev_ratio_direct, max_seg_direct, n_params_direct))
print(_row("SurrogateGLM (CART)", gini_surrogate, r2_surrogate, surrogate_report.metrics.deviance_ratio, max_seg_surrogate, n_params_surrogate))
print(_row("LassoGuidedGLM (CART+L1)", gini_lasso, r2_lasso, lasso_report.metrics.deviance_ratio, max_seg_lasso, n_params_lasso))
print()
print("  MaxSeg = max relative deviation on driver_age x region 2-way grid.")
print("  Gini% = fraction of GBM Gini retained.")
print("  R²GBM = how faithfully the GLM reproduces GBM training predictions.")

# ---------------------------------------------------------------------------
# 9. Interpretation
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print()
print(f"  GBM Gini (ceiling):                 {gini_gbm:.4f}")
print()
print(f"  SurrogateGLM retains:               {gini_surrogate/gini_gbm:.1%} of GBM Gini")
print(f"  LassoGuidedGLM retains:             {gini_lasso/gini_gbm:.1%} of GBM Gini")
print(f"  Direct GLM retains:                 {gini_direct/gini_gbm:.1%} of GBM Gini")
print()
print(f"  The fidelity R² tells a cleaner story than holdout Gini because it")
print(f"  measures how well the GLM copies the GBM's signal, independent of")
print(f"  claim noise in the holdout.")
print()
print(f"  Fidelity R² (training, vs GBM predictions):")
print(f"    Direct GLM:      {r2_direct:.4f}")
print(f"    SurrogateGLM:    {r2_surrogate:.4f}")
print(f"    LassoGuidedGLM:  {r2_lasso:.4f}")
print()
print(f"  MAE vs true claim rate (holdout):")
print(f"    CatBoost GBM:    {mae_gbm:.5f}")
print(f"    SurrogateGLM:    {mae_surrogate:.5f}")
print(f"    LassoGuidedGLM:  {mae_lasso:.5f}")
print(f"    Direct GLM:      {mae_direct:.5f}")
print()
print(f"  Max segment deviation on driver_age x region grid:")
print(f"    Direct GLM:      {max_seg_direct:.1%}  (quantile bins miss the true age curve)")
print(f"    SurrogateGLM:    {max_seg_surrogate:.1%}  (CART bins match GBM response shape)")
print(f"    LassoGuidedGLM:  {max_seg_lasso:.1%}")
print()
print(f"  Parameter counts:")
print(f"    Direct GLM:      {n_params_direct}")
print(f"    SurrogateGLM:    {n_params_surrogate}")
print(f"    LassoGuidedGLM:  {n_params_lasso}  ({n_zero_coeffs} lasso-zeroed)")
print()

# Diagnostic checks
if r2_surrogate > r2_direct:
    print(f"  PASS: SurrogateGLM has higher fidelity R² than direct GLM")
    print(f"        ({r2_surrogate:.4f} vs {r2_direct:.4f}) — distillation captures more GBM signal")
else:
    print(f"  INFO: Direct GLM has higher fidelity R² — the DGP may be too")
    print(f"        linear for CART binning to add value on this dataset")

if max_seg_surrogate < max_seg_direct:
    print(f"  PASS: SurrogateGLM has lower max segment deviation than direct GLM")
    print(f"        ({max_seg_surrogate:.1%} vs {max_seg_direct:.1%}) — factor tables are more faithful")
else:
    print(f"  INFO: Direct GLM has lower segment deviation — CART bins may have")
    print(f"        created too many levels for this dataset size")

print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
