"""Tests for _binning.py - OptimalBinner."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_distill._binning import OptimalBinner, _edges_to_spec, _format_bin_label


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_format_bin_label_normal():
    assert _format_bin_label(10.0, 25.0) == "[10.00, 25.00)"


def test_format_bin_label_inf_lower():
    label = _format_bin_label(-np.inf, 25.0)
    assert label == "[-inf, 25.00)"


def test_format_bin_label_inf_upper():
    label = _format_bin_label(25.0, np.inf)
    assert label == "[25.00, +inf)"


def test_edges_to_spec_produces_correct_count():
    spec = _edges_to_spec("x", [10.0, 20.0, 30.0], "tree")
    assert spec.n_bins == 4
    assert len(spec.bin_labels) == 4
    assert len(spec.bin_edges) == 5  # -inf, 10, 20, 30, +inf


def test_edges_to_spec_no_cuts_single_bin():
    spec = _edges_to_spec("x", [], "quantile")
    assert spec.n_bins == 1
    assert spec.bin_labels == ["[-inf, +inf)"]


# ---------------------------------------------------------------------------
# OptimalBinner - tree method
# ---------------------------------------------------------------------------


def test_tree_binner_basic(rng):
    x = rng.uniform(18, 80, 500)
    # Clear step-function target: y = 1 if x < 40 else 2
    y = np.where(x < 40, 1.0, 2.0)

    binner = OptimalBinner(max_bins=5, method="tree")
    spec = binner.fit_feature(x, y, "driver_age")

    # Should find the split near 40
    assert spec.n_bins >= 2
    interior_cuts = [e for e in spec.bin_edges if not np.isinf(e)]
    assert any(35 < c < 45 for c in interior_cuts), (
        f"Expected a cut near 40, got cuts: {interior_cuts}"
    )


def test_tree_binner_respects_max_bins(rng):
    x = rng.uniform(0, 100, 1000)
    y = rng.standard_normal(1000)
    binner = OptimalBinner(max_bins=4, method="tree")
    spec = binner.fit_feature(x, y, "x")
    assert spec.n_bins <= 4


def test_tree_binner_with_weights(rng):
    x = rng.uniform(18, 80, 500)
    y = np.where(x < 40, 1.0, 2.0)
    w = rng.uniform(0.1, 1.0, 500)

    binner = OptimalBinner(max_bins=5, method="tree")
    spec = binner.fit_feature(x, y, "driver_age", weights=w)
    assert spec.n_bins >= 2


# ---------------------------------------------------------------------------
# OptimalBinner - quantile method
# ---------------------------------------------------------------------------


def test_quantile_binner_basic(rng):
    x = rng.uniform(0, 100, 500)
    y = rng.standard_normal(500)
    binner = OptimalBinner(max_bins=5, method="quantile")
    spec = binner.fit_feature(x, y, "x")
    assert spec.n_bins <= 5
    assert spec.method == "quantile"


def test_quantile_binner_low_cardinality():
    # Only 3 distinct values
    x = np.array([1.0, 2.0, 3.0] * 100)
    y = np.random.default_rng(0).standard_normal(300)
    binner = OptimalBinner(max_bins=10, method="quantile")
    spec = binner.fit_feature(x, y, "x")
    assert spec.n_bins <= 3


# ---------------------------------------------------------------------------
# OptimalBinner - isotonic method
# ---------------------------------------------------------------------------


def test_isotonic_binner_monotone(rng):
    x = rng.uniform(0, 100, 500)
    # Monotone increasing response
    y = 0.01 * x + rng.standard_normal(500) * 0.1

    binner = OptimalBinner(max_bins=8, method="isotonic")
    spec = binner.fit_feature(x, y, "x")
    # Should produce at least 1 bin
    assert spec.n_bins >= 1
    assert spec.method == "isotonic"


# ---------------------------------------------------------------------------
# OptimalBinner - batch fit
# ---------------------------------------------------------------------------


def test_batch_fit(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    preds = rng.uniform(0.05, 0.3, len(X))

    binner = OptimalBinner(max_bins=6)
    specs = binner.fit(X, preds, features=["driver_age", "vehicle_value", "ncd_years"])

    assert set(specs.keys()) == {"driver_age", "vehicle_value", "ncd_years"}
    for feat, spec in specs.items():
        assert spec.n_bins >= 1
        assert spec.n_bins <= 6


def test_batch_fit_missing_feature_raises(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    preds = rng.uniform(0.05, 0.3, len(X))
    binner = OptimalBinner(max_bins=6)
    with pytest.raises(ValueError, match="not found"):
        binner.fit(X, preds, features=["nonexistent_column"])


def test_batch_fit_method_overrides(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    preds = rng.uniform(0.05, 0.3, len(X))

    binner = OptimalBinner(max_bins=6, method="tree")
    specs = binner.fit(
        X,
        preds,
        features=["driver_age", "vehicle_value"],
        method_overrides={"vehicle_value": "quantile"},
    )
    assert specs["driver_age"].method == "tree"
    assert specs["vehicle_value"].method == "quantile"


# ---------------------------------------------------------------------------
# OptimalBinner - transform
# ---------------------------------------------------------------------------


def test_transform_adds_bin_columns(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    preds = rng.uniform(0.05, 0.3, len(X))
    binner = OptimalBinner(max_bins=5)
    specs = binner.fit(X, preds, features=["driver_age", "vehicle_value"])
    X_binned = binner.transform(X, specs)

    assert "driver_age__bin" in X_binned.columns
    assert "vehicle_value__bin" in X_binned.columns
    # Original numeric columns preserved
    assert "driver_age" in X_binned.columns


def test_transform_no_nulls(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    preds = rng.uniform(0.05, 0.3, len(X))
    binner = OptimalBinner(max_bins=5)
    specs = binner.fit(X, preds, features=["driver_age"])
    X_binned = binner.transform(X, specs)
    assert X_binned["driver_age__bin"].null_count() == 0


def test_transform_bin_count_matches_spec(rng, synthetic_motor_data):
    X = synthetic_motor_data["X"]
    preds = rng.uniform(0.05, 0.3, len(X))
    binner = OptimalBinner(max_bins=5)
    specs = binner.fit(X, preds, features=["driver_age"])
    X_binned = binner.transform(X, specs)
    n_distinct_bins = X_binned["driver_age__bin"].n_unique()
    assert n_distinct_bins <= specs["driver_age"].n_bins
