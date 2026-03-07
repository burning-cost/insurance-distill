"""Tests for _validation.py - Gini, deviance ratio, segment deviation."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_distill._validation import (
    compute_gini,
    compute_deviance_ratio,
    compute_segment_deviation,
    double_lift_chart,
)


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------


def test_gini_perfect_predictions():
    """Perfect rank order -> Gini = 1.0."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gini = compute_gini(y)
    assert gini > 0.9


def test_gini_random_predictions(rng):
    """Random predictions -> Gini near 0."""
    y = rng.uniform(0, 1, 10_000)
    gini = compute_gini(y)
    assert gini < 0.1


def test_gini_constant_predictions():
    """Constant predictions -> Gini = 0."""
    y = np.full(100, 0.5)
    gini = compute_gini(y)
    assert gini == pytest.approx(0.0, abs=1e-6)


def test_gini_nonnegative(rng):
    y = rng.exponential(1.0, 1000)
    gini = compute_gini(y)
    assert 0.0 <= gini <= 1.0


def test_gini_with_weights(rng):
    y = rng.exponential(1.0, 500)
    w = rng.uniform(0.1, 1.0, 500)
    gini = compute_gini(y, weights=w)
    assert 0.0 <= gini <= 1.0


def test_gini_empty():
    gini = compute_gini(np.array([]))
    assert gini == 0.0


# ---------------------------------------------------------------------------
# Deviance ratio
# ---------------------------------------------------------------------------


def test_deviance_ratio_perfect_poisson():
    """GLM that perfectly replicates pseudo -> deviance ratio = 1.0."""
    pseudo = np.exp(np.random.default_rng(0).normal(0, 0.5, 500))
    ratio = compute_deviance_ratio(pseudo, pseudo, "poisson")
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_deviance_ratio_null_model_poisson():
    """Null model (constant prediction) -> deviance ratio = 0.0."""
    rng = np.random.default_rng(1)
    pseudo = rng.exponential(1.0, 500)
    null_pred = np.full_like(pseudo, pseudo.mean())
    ratio = compute_deviance_ratio(pseudo, null_pred, "poisson")
    assert ratio == pytest.approx(0.0, abs=0.01)


def test_deviance_ratio_gamma_perfect():
    pseudo = np.exp(np.random.default_rng(2).normal(0, 0.5, 500))
    ratio = compute_deviance_ratio(pseudo, pseudo, "gamma")
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_deviance_ratio_unknown_family():
    with pytest.raises(ValueError, match="Unknown family"):
        compute_deviance_ratio(np.ones(10), np.ones(10), "normal")


def test_deviance_ratio_between_0_and_1(rng):
    pseudo = rng.exponential(1.0, 500)
    # Some noise on the GLM prediction
    glm = pseudo * rng.uniform(0.8, 1.2, 500)
    ratio = compute_deviance_ratio(pseudo, glm, "poisson")
    assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# Segment deviation
# ---------------------------------------------------------------------------


def test_segment_deviation_perfect(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    pseudo = rng.uniform(0.05, 0.3, len(X))

    from insurance_distill._binning import OptimalBinner

    binner = OptimalBinner(max_bins=5)
    specs = binner.fit(X, pseudo, features=["driver_age"])
    X_binned = binner.transform(X, specs)

    max_dev, mean_dev, n_seg = compute_segment_deviation(
        X_binned=X_binned,
        pseudo=pseudo,
        glm_pred=pseudo,  # perfect match
        exposure=np.ones(len(X)),
        bin_features=["driver_age__bin"],
        cat_features=[],
    )
    assert max_dev == pytest.approx(0.0, abs=1e-6)
    assert mean_dev == pytest.approx(0.0, abs=1e-6)


def test_segment_deviation_no_groups(rng):
    n = 100
    pseudo = rng.uniform(0.1, 0.5, n)
    glm = pseudo * 1.1  # 10% bias

    X_dummy = pl.DataFrame({"x": np.arange(n, dtype=float)})
    max_dev, mean_dev, n_seg = compute_segment_deviation(
        X_binned=X_dummy,
        pseudo=pseudo,
        glm_pred=glm,
        exposure=np.ones(n),
        bin_features=[],
        cat_features=[],
    )
    assert n_seg == 1
    assert max_dev > 0.0


def test_segment_deviation_with_cat(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    pseudo = rng.uniform(0.05, 0.3, len(X))
    glm = pseudo * rng.uniform(0.9, 1.1, len(X))

    from insurance_distill._binning import OptimalBinner

    binner = OptimalBinner(max_bins=5)
    specs = binner.fit(X, pseudo, features=["driver_age"])
    X_binned = binner.transform(X, specs)

    max_dev, mean_dev, n_seg = compute_segment_deviation(
        X_binned=X_binned,
        pseudo=pseudo,
        glm_pred=glm,
        exposure=synthetic_motor_data["exposure"],
        bin_features=["driver_age__bin"],
        cat_features=["region"],
    )
    assert n_seg > 1
    assert 0.0 <= max_dev <= 1.0


# ---------------------------------------------------------------------------
# Double-lift chart
# ---------------------------------------------------------------------------


def test_double_lift_chart_shape(rng):
    n = 1000
    pseudo = rng.exponential(1.0, n)
    glm = pseudo * rng.uniform(0.8, 1.2, n)
    chart = double_lift_chart(pseudo, glm, n_deciles=10)
    assert "decile" in chart.columns
    assert "avg_gbm" in chart.columns
    assert "avg_glm" in chart.columns
    assert "ratio_gbm_to_glm" in chart.columns
    assert len(chart) <= 10


def test_double_lift_chart_perfect_agreement(rng):
    """If GBM and GLM agree perfectly, all ratios should be ~1.0."""
    n = 1000
    pseudo = rng.exponential(1.0, n)
    chart = double_lift_chart(pseudo, pseudo, n_deciles=10)
    ratios = chart["ratio_gbm_to_glm"].to_numpy()
    np.testing.assert_allclose(ratios, 1.0, atol=1e-6)
