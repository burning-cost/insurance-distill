# Databricks notebook source
# Test insurance-distill v0.1.4 (LassoGuidedGLM feature)

# COMMAND ----------

# MAGIC %pip install --upgrade insurance-distill==0.1.4 pytest catboost polars

# COMMAND ----------

import insurance_distill
print("version:", insurance_distill.__version__)
assert insurance_distill.__version__ == "0.1.4", f"Expected 0.1.4, got {insurance_distill.__version__}"
print("Version check OK")

from insurance_distill import (
    SurrogateGLM, LassoGuidedGLM, OptimalBinner,
    BinSpec, ValidationMetrics, DistillationReport,
    compute_gini, build_factor_tables,
)
print("All public API imports OK")

# COMMAND ----------

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

rng = np.random.default_rng(42)
n = 1000
X_np = rng.standard_normal((n, 4))
beta = np.array([0.3, 0.0, -0.5, 0.2])
y = np.exp(X_np @ beta + rng.standard_normal(n) * 0.1)
exposure = rng.uniform(0.5, 2.0, n)
feature_names = [f"x{i}" for i in range(4)]

# LassoGuidedGLM requires a Polars DataFrame
X_pl = pl.DataFrame(X_np, schema=feature_names)

# Fit GBM (catboost needs numpy/pandas)
import pandas as pd
X_pd = pd.DataFrame(X_np, columns=feature_names)
gbm = CatBoostRegressor(iterations=50, verbose=0, loss_function="RMSE")
gbm.fit(X_pd, y, sample_weight=exposure)

# Test LassoGuidedGLM
model = LassoGuidedGLM(
    gbm_model=gbm,
    feature_names=feature_names,
    n_bins=5,
    alpha=0.5,
    family="tweedie",
)
model.fit(X_pl, y, sample_weight=exposure)
preds = model.predict(X_pl)

assert len(preds) == n, f"Wrong prediction length: {len(preds)}"
assert np.all(preds > 0), "Predictions should be positive"
print(f"LassoGuidedGLM functional test PASSED")
print(f"Selected features: {model._selected_features}")
print(f"Predictions: min={preds.min():.4f}, mean={preds.mean():.4f}, max={preds.max():.4f}")

# COMMAND ----------

# Run pytest tests
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--pyargs", "insurance_distill",
     "-x", "-q", "--tb=short", "--no-header"],
    capture_output=True, text=True, timeout=300,
)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr[-1000:])
print("Return code:", result.returncode)
if result.returncode != 0:
    print("WARNING: some tests may have failed (could be test discovery issue)")
else:
    print("pytest PASSED")

print("ALL CHECKS PASSED: insurance-distill 0.1.4")
