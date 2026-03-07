# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-distill: Motor Frequency Distillation Demo
# MAGIC
# MAGIC **Problem**: We have a CatBoost frequency model that outperforms our GLM, but our
# MAGIC rating engine needs multiplicative factor tables.
# MAGIC
# MAGIC **Solution**: Use `insurance-distill` to fit a Poisson GLM surrogate on the CatBoost
# MAGIC predictions, validate it, and export factor tables as CSVs.
# MAGIC
# MAGIC This notebook runs end-to-end on synthetic UK motor data and demonstrates
# MAGIC the full workflow including validation charts.

# COMMAND ----------

# MAGIC %pip install insurance-distill "catboost>=1.2" matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor insurance data

# COMMAND ----------

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

rng = np.random.default_rng(2024)
n = 50_000

# Feature simulation
driver_age = rng.uniform(17, 80, n)
vehicle_value = rng.uniform(3_000, 80_000, n)
ncd_years = rng.integers(0, 11, n).astype(float)
vehicle_age = rng.uniform(0, 20, n)
annual_mileage = rng.integers(3_000, 30_000, n).astype(float)
region = rng.choice(["London", "South East", "North", "Midlands", "Scotland"], n)
exposure = rng.uniform(0.1, 1.0, n)

# Non-linear frequency model (ground truth)
log_mu = (
    -3.5
    + np.where(driver_age < 25, 0.8 * (25 - driver_age) / 7, 0.0)   # young driver
    + np.where(driver_age > 65, 0.3 * (driver_age - 65) / 15, 0.0)  # elderly driver
    - 0.07 * ncd_years                                                 # NCD
    + 0.000005 * vehicle_value                                        # value
    + 0.012 * vehicle_age                                             # vehicle age
    + np.log(annual_mileage / 10_000) * 0.3                          # mileage
    + np.where(region == "London", 0.35,
      np.where(region == "South East", 0.15,
      np.where(region == "North", -0.05,
      np.where(region == "Scotland", -0.20, 0.0))))                   # region
)
mu = exposure * np.exp(log_mu)
y = rng.poisson(mu).astype(float)

print(f"Rows: {n:,}")
print(f"Overall frequency: {(y / exposure).mean():.4f}")
print(f"Claim count: {y.sum():.0f}")

X = pl.DataFrame({
    "driver_age": driver_age,
    "vehicle_value": vehicle_value,
    "ncd_years": ncd_years,
    "vehicle_age": vehicle_age,
    "annual_mileage": annual_mileage,
    "region": region,
})

print(X.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit a CatBoost frequency model

# COMMAND ----------

from sklearn.model_selection import train_test_split

idx = np.arange(n)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
exp_train = exposure[train_idx]
exp_test = exposure[test_idx]

# CatBoost treats strings as categoricals automatically
cb_model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=5,
    loss_function="Poisson",
    cat_features=["region"],
    random_seed=42,
    verbose=50,
)

# For Poisson, we model rate = y / exposure, then use exposure as weight
rate_train = np.clip(y_train / exp_train, 0, None)

cb_model.fit(
    X_train.to_pandas(),
    rate_train,
    sample_weight=exp_train,
)

# Training set Gini
from insurance_distill import compute_gini
cb_preds_train = cb_model.predict(X_train.to_pandas())
gini_cb = compute_gini(cb_preds_train, exp_train)
print(f"CatBoost Gini (train): {gini_cb:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Distil into a surrogate GLM

# COMMAND ----------

from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=cb_model,
    X_train=X_train,
    y_train=y_train,
    exposure=exp_train,
    family="poisson",
)

surrogate.fit(
    features=["driver_age", "vehicle_value", "ncd_years", "vehicle_age", "annual_mileage"],
    categorical_features=["region"],
    max_bins=10,
    binning_method="tree",
    method_overrides={
        "ncd_years": "isotonic",   # NCD should be monotone
    },
)

print("Surrogate GLM fitted successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validate the surrogate

# COMMAND ----------

report = surrogate.report()
print(report.metrics.summary())
print()
print(repr(report))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Double-lift chart
# MAGIC
# MAGIC This shows GBM vs GLM predictions across deciles of the ratio GBM/GLM.
# MAGIC Decile 1 = segments where GBM is much cheaper than GLM;
# MAGIC Decile 10 = segments where GBM is much more expensive than GLM.

# COMMAND ----------

import matplotlib.pyplot as plt

lift = report.lift_chart
fig, ax = plt.subplots(figsize=(10, 5))
x = lift["decile"].to_list()
ax.plot(x, lift["avg_gbm"].to_list(), "o-", label="CatBoost", color="#1f77b4", linewidth=2)
ax.plot(x, lift["avg_glm"].to_list(), "s--", label="GLM surrogate", color="#ff7f0e", linewidth=2)
ax.set_xlabel("Decile (sorted by GBM/GLM ratio)")
ax.set_ylabel("Average predicted frequency")
ax.set_title("Double-lift chart: CatBoost vs GLM surrogate")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect factor tables

# COMMAND ----------

# Driver age - the most important variable for motor pricing
age_table = surrogate.factor_table("driver_age")
print("Driver age factor table:")
print(age_table)

# COMMAND ----------

# Vehicle value
val_table = surrogate.factor_table("vehicle_value")
print("Vehicle value factor table:")
print(val_table)

# COMMAND ----------

# NCD (should be monotone decreasing due to isotonic binning)
ncd_table = surrogate.factor_table("ncd_years")
print("NCD years factor table:")
print(ncd_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot relativities for driver age

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, feat in zip(axes, ["driver_age", "ncd_years", "vehicle_value"]):
    tbl = surrogate.factor_table(feat)
    levels = tbl["level"].to_list()
    rels = tbl["relativity"].to_numpy()
    x = np.arange(len(levels))
    ax.bar(x, rels, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relativity")
    ax.set_title(feat.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Export factor tables to CSV (Radar/Emblem-compatible)

# COMMAND ----------

import os

output_dir = "/tmp/motor_freq_factors"
written = surrogate.export_csv(output_dir, prefix="motor_freq_")

print("Written files:")
for path in written:
    size = os.path.getsize(path)
    print(f"  {path} ({size} bytes)")

# Preview the driver_age CSV
print()
print("--- motor_freq_driver_age.csv ---")
print(open(os.path.join(output_dir, "motor_freq_driver_age.csv")).read())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. All GLM coefficients

# COMMAND ----------

print("All fitted GLM coefficients:")
print(report.glm_coefficients.sort("log_coefficient", descending=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Radar CSV format
# MAGIC
# MAGIC Use `format_radar_csv()` to get a Radar-compatible string for manual
# MAGIC inspection or upload to the rating system.

# COMMAND ----------

from insurance_distill import format_radar_csv

radar_csv = format_radar_csv(surrogate.factor_table("driver_age"), "driver_age")
print(radar_csv)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The distillation pipeline:
# MAGIC 1. Fitted a CatBoost model on 40,000 synthetic motor policies
# MAGIC 2. Used `insurance-distill` to fit a Poisson GLM surrogate in under 10 seconds
# MAGIC 3. Validated: Gini ratio and segment deviations
# MAGIC 4. Exported one CSV per rating variable, ready for Radar/Emblem
# MAGIC
# MAGIC The GLM surrogate retains the majority of the GBM's discriminatory power
# MAGIC while being directly loadable into a multiplicative rating engine.
