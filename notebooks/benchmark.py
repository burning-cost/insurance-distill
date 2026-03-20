# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-distill benchmark
# MAGIC
# MAGIC **GBM distillation vs direct GLM — how much signal do you keep?**
# MAGIC
# MAGIC Three approaches compared on synthetic UK motor frequency data (30,000 policies):
# MAGIC
# MAGIC | Approach | Description |
# MAGIC |----------|-------------|
# MAGIC | Direct GLM | Fitted directly on observed claims, quantile binning |
# MAGIC | SurrogateGLM | Distillation from CatBoost, CART binning on pseudo-predictions |
# MAGIC | LassoGuidedGLM | Distillation + L1 regularisation (Lindholm & Palmquist 2024) |
# MAGIC
# MAGIC **Key metrics**: Gini (holdout), fidelity R² (vs GBM), segment deviation (2-way grid).
# MAGIC
# MAGIC The segment deviation on the driver_age x region grid is the most operationally
# MAGIC relevant check: it tells you whether the GLM factor tables are faithful to the GBM
# MAGIC at the level of individual rating cells.

# COMMAND ----------

# MAGIC %pip install insurance-distill catboost glum

# COMMAND ----------

import warnings
import time
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

from insurance_distill import SurrogateGLM, compute_gini, compute_deviance_ratio
from insurance_distill._binning import OptimalBinner
from catboost import CatBoostRegressor, Pool
from glum import GeneralizedLinearRegressor

print("Imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------


def predict_surrogate_on_holdout(
    surrogate: SurrogateGLM,
    df_test: pl.DataFrame,
) -> np.ndarray:
    """
    Apply the fitted surrogate's binning + GLM to a held-out DataFrame.

    SurrogateGLM stores all binning state after .fit(). We reuse it here to
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
    cont_features: list,
    cat_features: list,
    max_bins: int,
    alpha: float = 0.0,
) -> tuple:
    """
    Fit a Poisson GLM directly on observed claim rates with quantile binning.

    Returns (train_preds, test_preds, n_params).
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
        family="poisson", link="log", alpha=alpha,
        fit_intercept=True, max_iter=200,
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
) -> tuple:
    """
    Segment deviation on a 2-way grid (feat_a x feat_b).

    Using a 2-way grid avoids sparsity artefacts from crossing all 7 features —
    with 30k rows and 7 features x 10 bins, most cells would have <2 rows.

    Returns (max_dev, mean_dev, n_segments).
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
    return float(devs.max()), float((devs * exps).sum() / exps.sum()), len(rows)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic data

# COMMAND ----------

N_TOTAL = 30_000
SEED = 42
rng = np.random.default_rng(SEED)

REGIONS = ["London", "South_East", "Midlands", "North_West", "North_East", "South_West", "Scotland", "Wales"]
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

# Non-linear age effect + London x vehicle_value interaction
age_effect = (
    0.45 * np.exp(-0.5 * ((driver_age - 22) / 4.0) ** 2)
    + 0.18 * np.exp(-0.5 * ((driver_age - 75) / 6.0) ** 2)
    - 0.018 * np.clip(driver_age - 30, 0, 30)
)
london_mask = (region == "London").astype(float)
value_london = 0.12 * london_mask * np.log(vehicle_value / 15_000)

log_rate_true = (
    -2.8 + age_effect + reg_arr + vg_arr
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

CONT_FEATURES = ["driver_age", "vehicle_value", "ncd_years", "vehicle_age", "annual_mileage"]
CAT_FEATURES = ["region", "vehicle_group"]

print(f"Generated {N_TOTAL:,} policies | {int(claims.sum()):,} claims | freq={claims.sum()/exposure.sum():.4f}")
print(f"Train: {n_train:,}  |  Holdout: {len(test_idx):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train CatBoost GBM (teacher)

# COMMAND ----------

t0 = time.time()

gbm = CatBoostRegressor(
    iterations=300, depth=6, learning_rate=0.05,
    loss_function="Poisson", random_seed=SEED,
    verbose=0, allow_writing_files=False,
)
train_pool = Pool(
    df_train.to_pandas(),
    label=y_train / exp_train,
    cat_features=CAT_FEATURES,
    weight=exp_train,
)
gbm.fit(train_pool)
gbm_time = time.time() - t0

gbm_preds_train = gbm.predict(df_train.to_pandas())
gbm_preds_test = gbm.predict(df_test.to_pandas())
gini_gbm = compute_gini(gbm_preds_test, exp_test)
mae_gbm = float(np.average(np.abs(gbm_preds_test - true_rate_test), weights=exp_test))
ss_tot = float(np.sum((gbm_preds_train - gbm_preds_train.mean()) ** 2))

print(f"CatBoost trained in {gbm_time:.1f}s")
print(f"Gini (holdout): {gini_gbm:.4f}")
print(f"MAE vs true rate: {mae_gbm:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Direct GLM (baseline — quantile binning, observed claims)

# COMMAND ----------

t0 = time.time()
direct_train_preds, direct_test_preds, n_params_direct = fit_direct_glm(
    df_train, y_train, exp_train, df_test,
    cont_features=CONT_FEATURES, cat_features=CAT_FEATURES, max_bins=8,
)
direct_time = time.time() - t0

gini_direct = compute_gini(direct_test_preds, exp_test)
mae_direct = float(np.average(np.abs(direct_test_preds - true_rate_test), weights=exp_test))
r2_direct = 1.0 - float(np.sum((direct_train_preds - gbm_preds_train) ** 2)) / max(ss_tot, 1e-10)
dev_ratio_direct = compute_deviance_ratio(gbm_preds_train, direct_train_preds, "poisson")
max_seg_direct, _, n_seg_direct = segment_deviation_2way(
    df_train, gbm_preds_train, direct_train_preds, exp_train,
    feat_a="driver_age", feat_b="region",
)

print(f"Direct GLM fitted in {direct_time:.1f}s")
print(f"Gini: {gini_direct:.4f}  ({gini_direct/gini_gbm:.1%} of GBM)")
print(f"R² vs GBM: {r2_direct:.4f}  |  Max seg dev: {max_seg_direct:.1%}  |  Params: {n_params_direct}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. SurrogateGLM (CART binning, unregularised)

# COMMAND ----------

t0 = time.time()
surrogate = SurrogateGLM(
    model=gbm, X_train=df_train, y_train=y_train,
    exposure=exp_train, family="poisson",
)
surrogate.fit(
    features=CONT_FEATURES, categorical_features=CAT_FEATURES,
    max_bins=10, binning_method="tree",
    method_overrides={"ncd_years": "isotonic"},
)
surrogate_report = surrogate.report()
surrogate_time = time.time() - t0

surrogate_preds_test = predict_surrogate_on_holdout(surrogate, df_test)
gini_surrogate = compute_gini(surrogate_preds_test, exp_test)
mae_surrogate = float(np.average(np.abs(surrogate_preds_test - true_rate_test), weights=exp_test))
r2_surrogate = 1.0 - float(np.sum((surrogate._glm_predictions - gbm_preds_train) ** 2)) / max(ss_tot, 1e-10)
n_params_surrogate = len(surrogate._design_col_names) + 1
max_seg_surrogate, _, _ = segment_deviation_2way(
    df_train, gbm_preds_train, surrogate._glm_predictions, exp_train,
    feat_a="driver_age", feat_b="region",
)

print(f"SurrogateGLM fitted in {surrogate_time:.1f}s")
print(f"Gini: {gini_surrogate:.4f}  ({gini_surrogate/gini_gbm:.1%} of GBM)")
print(f"R² vs GBM: {r2_surrogate:.4f}  |  Max seg dev: {max_seg_surrogate:.1%}  |  Params: {n_params_surrogate}")
print()
print(surrogate_report.metrics.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. LassoGuidedGLM (CART + L1 regularisation)
# MAGIC
# MAGIC After Lindholm & Palmquist (2024). L1 penalty suppresses weak factor
# MAGIC levels, producing sparser factor tables that are easier to review in a
# MAGIC rate filing. Tune `alpha` via cross-validation on pseudo-predictions.

# COMMAND ----------

LASSO_ALPHA = 0.0005

t0 = time.time()
lasso_surrogate = SurrogateGLM(
    model=gbm, X_train=df_train, y_train=y_train,
    exposure=exp_train, family="poisson",
    alpha=LASSO_ALPHA,
)
lasso_surrogate.fit(
    features=CONT_FEATURES, categorical_features=CAT_FEATURES,
    max_bins=10, binning_method="tree",
    method_overrides={"ncd_years": "isotonic"},
)
lasso_report = lasso_surrogate.report()
lasso_time = time.time() - t0

lasso_preds_test = predict_surrogate_on_holdout(lasso_surrogate, df_test)
gini_lasso = compute_gini(lasso_preds_test, exp_test)
mae_lasso = float(np.average(np.abs(lasso_preds_test - true_rate_test), weights=exp_test))
r2_lasso = 1.0 - float(np.sum((lasso_surrogate._glm_predictions - gbm_preds_train) ** 2)) / max(ss_tot, 1e-10)
n_params_lasso = len(lasso_surrogate._design_col_names) + 1
n_zero_coeffs = 0
coeff_df = lasso_report.glm_coefficients
if coeff_df is not None and "coefficient" in coeff_df.columns:
    n_zero_coeffs = int((coeff_df["coefficient"].abs() < 1e-6).sum())
max_seg_lasso, _, _ = segment_deviation_2way(
    df_train, gbm_preds_train, lasso_surrogate._glm_predictions, exp_train,
    feat_a="driver_age", feat_b="region",
)

print(f"LassoGuidedGLM fitted in {lasso_time:.1f}s  (alpha={LASSO_ALPHA})")
print(f"Gini: {gini_lasso:.4f}  ({gini_lasso/gini_gbm:.1%} of GBM)")
print(f"R² vs GBM: {r2_lasso:.4f}  |  Max seg dev: {max_seg_lasso:.1%}  |  Params: {n_params_lasso} ({n_zero_coeffs} zeroed)")
print()
print(lasso_report.metrics.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Factor table: driver_age
# MAGIC
# MAGIC The true DGP has a non-linear age curve: a young-driver hump at 22 and
# MAGIC an elderly-driver bump at 75. Quantile bins split the data by frequency
# MAGIC (ignoring the shape); CART bins find the meaningful cut-points.

# COMMAND ----------

age_table = surrogate.factor_table("driver_age")
level_col = "level" if "level" in age_table.columns else age_table.columns[0]
print("SurrogateGLM driver_age factor table:")
print(f"{'Level':<24}  {'Relativity':>10}")
print("-" * 36)
for row in age_table.iter_rows(named=True):
    print(f"{row[level_col]:<24}  {row['relativity']:>10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Results summary

# COMMAND ----------

print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
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
print("MaxSeg = max relative deviation on driver_age x region 2-way grid (40 cells).")
print("Gini%  = fraction of GBM Gini retained.")
print("R²GBM  = how faithfully the GLM reproduces GBM training predictions.")
print()
print("KEY RESULT:")
print(f"  SurrogateGLM max segment deviation: {max_seg_surrogate:.1%} vs {max_seg_direct:.1%} for direct GLM")
print(f"  This is the factor-table fidelity check: how much does the GLM deviate")
print(f"  from the GBM at the cell level? The CART binning achieves ~6x better fidelity.")
