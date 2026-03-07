"""
Shared test fixtures for insurance-distill.

We generate a small synthetic motor insurance dataset and fit a simple
sklearn model to use as the "GBM" in tests. This avoids any dependency
on CatBoost in the core test suite, keeping things fast and portable.

The synthetic data has the following structure:
- driver_age: continuous, 18-80
- vehicle_value: continuous, 5000-60000
- ncd_years: discrete, 0-10
- region: categorical, 4 regions
- exposure: continuous, 0.1-1.0
- freq: Poisson frequency (the actual claims target)
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_motor_data(rng: np.random.Generator) -> dict:
    """
    Generate a small synthetic motor insurance dataset.

    Returns a dict with keys: X (Polars DataFrame), y (np.ndarray),
    exposure (np.ndarray).
    """
    n = 2000
    driver_age = rng.uniform(18, 80, n)
    vehicle_value = rng.uniform(5_000, 60_000, n)
    ncd_years = rng.integers(0, 11, n).astype(float)
    region = rng.choice(["North", "South", "East", "West"], n)
    exposure = rng.uniform(0.1, 1.0, n)

    # Simulate a non-linear frequency: higher risk for young drivers and high-value cars
    log_mu = (
        -3.0
        - 0.04 * np.clip(driver_age - 25, 0, None)  # age effect
        + 0.01 * np.clip(25 - driver_age, 0, None)   # young driver surcharge
        + 0.00002 * vehicle_value                      # vehicle value effect
        - 0.08 * ncd_years                            # NCD discount
        + np.where(region == "North", 0.2, 0.0)       # region effect
    )
    mu = exposure * np.exp(log_mu)
    y = rng.poisson(mu).astype(float)

    X = pl.DataFrame(
        {
            "driver_age": driver_age,
            "vehicle_value": vehicle_value,
            "ncd_years": ncd_years,
            "region": region,
        }
    )

    return {"X": X, "y": y, "exposure": exposure, "log_mu": log_mu}


@pytest.fixture(scope="session")
def fitted_gbm(synthetic_motor_data: dict) -> GradientBoostingRegressor:
    """Fit a simple sklearn GBM on the synthetic data (numeric features only)."""
    data = synthetic_motor_data
    X_num = data["X"].select(["driver_age", "vehicle_value", "ncd_years"]).to_numpy()
    y = data["y"]
    exposure = data["exposure"]

    # For frequency, fit on claims / exposure as a rate
    rate = np.clip(y / exposure, 0, None)

    gbm = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    gbm.fit(X_num, rate, sample_weight=exposure)
    return gbm


@pytest.fixture(scope="session")
def fitted_gbm_with_region(synthetic_motor_data: dict):
    """
    A simple sklearn GBM that uses numeric features only (region excluded
    since sklearn GBM does not handle strings). The region column is kept
    in X_train for the surrogate to treat as a categorical feature.
    """
    data = synthetic_motor_data
    X_num = data["X"].select(["driver_age", "vehicle_value", "ncd_years"]).to_numpy()
    y = data["y"]
    exposure = data["exposure"]
    rate = np.clip(y / exposure, 0, None)

    gbm = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    gbm.fit(X_num, rate, sample_weight=exposure)
    return gbm


class NumericOnlyWrapper:
    """
    Wraps a GBM trained on numeric columns only, but accepts a Polars
    DataFrame with extra columns. Used to test that SurrogateGLM can
    handle models that ignore some columns.
    """

    def __init__(self, gbm, numeric_cols: list[str]) -> None:
        self._gbm = gbm
        self._cols = numeric_cols

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._gbm.predict(X.select(self._cols).to_numpy())
