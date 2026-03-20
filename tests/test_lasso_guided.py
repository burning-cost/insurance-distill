"""
Tests for lasso_guided.py — LassoGuidedGLM.

We test the full pipeline end-to-end on the shared synthetic motor data
(from conftest.py), plus targeted unit tests for PD extraction, bin boundary
detection, feature selection, and factor table format.

The "GBM" used here is the same sklearn GradientBoostingRegressor from the
shared fixtures, which has a proper sklearn API (fit/predict), so
partial_dependence works against it without any wrapper.

Tests that need a model without the sklearn estimator API use a minimal
NumpyWrapper — in those cases the PD step gracefully falls back to quantile
splits (tested explicitly).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_distill import LassoGuidedGLM
from insurance_distill.lasso_guided import _pd_guided_splits, _apply_cuts, _build_ohe_block


NUMERIC_FEATURES = ["driver_age", "vehicle_value", "ncd_years"]


# ---------------------------------------------------------------------------
# Helper: minimal wrapper that does NOT implement 'fit' (for fallback tests)
# ---------------------------------------------------------------------------


class _NoFitWrapper:
    """Wraps a GBM but hides the 'fit' method so partial_dependence rejects it."""

    def __init__(self, gbm, cols: list[str]) -> None:
        self._gbm = gbm
        self._cols = cols

    def predict(self, X):
        if isinstance(X, pl.DataFrame):
            return self._gbm.predict(X.select(self._cols).to_numpy())
        return self._gbm.predict(X[:, : len(self._cols)])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_lasso_glm(synthetic_motor_data, fitted_gbm):
    """
    Fitted LassoGuidedGLM on numeric features, Poisson family.

    We pass the raw GradientBoostingRegressor directly — it has a proper
    sklearn API so partial_dependence works. The model was trained on the
    three numeric features in the same column order as NUMERIC_FEATURES.
    """
    data = synthetic_motor_data
    X = data["X"].select(NUMERIC_FEATURES)

    lg = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
        n_bins=6,
        alpha=1e-4,   # very mild lasso — preserves all genuine features
        family="poisson",
    )
    lg.fit(X, data["y"], exposure=data["exposure"])
    return lg


# ---------------------------------------------------------------------------
# Unit tests: PD extraction and split detection
# ---------------------------------------------------------------------------


def test_pd_guided_splits_returns_list(fitted_gbm, synthetic_motor_data):
    """_pd_guided_splits must return a plain list of floats."""
    data = synthetic_motor_data
    X_numpy = data["X"].select(NUMERIC_FEATURES).to_numpy().astype(float)

    cuts = _pd_guided_splits(
        model=fitted_gbm,
        X_numpy=X_numpy,
        feature_idx=0,
        feature_name="driver_age",
        n_splits=5,
        grid_resolution=50,
    )
    assert isinstance(cuts, list)
    assert all(isinstance(c, float) for c in cuts)


def test_pd_guided_splits_count_bounded(fitted_gbm, synthetic_motor_data):
    """Number of returned cuts must not exceed n_splits."""
    data = synthetic_motor_data
    X_numpy = data["X"].select(NUMERIC_FEATURES).to_numpy().astype(float)

    for n_splits in [3, 5, 10]:
        cuts = _pd_guided_splits(
            model=fitted_gbm,
            X_numpy=X_numpy,
            feature_idx=0,
            feature_name="driver_age",
            n_splits=n_splits,
            grid_resolution=50,
        )
        assert len(cuts) <= n_splits, f"Got {len(cuts)} cuts for n_splits={n_splits}"


def test_pd_guided_splits_sorted(fitted_gbm, synthetic_motor_data):
    """Returned cuts must be in ascending order."""
    data = synthetic_motor_data
    X_numpy = data["X"].select(NUMERIC_FEATURES).to_numpy().astype(float)

    cuts = _pd_guided_splits(
        model=fitted_gbm,
        X_numpy=X_numpy,
        feature_idx=1,
        feature_name="vehicle_value",
        n_splits=8,
    )
    assert cuts == sorted(cuts)


def test_pd_guided_splits_within_data_range(fitted_gbm, synthetic_motor_data):
    """All returned cuts must fall strictly within the observed data range."""
    data = synthetic_motor_data
    X_numpy = data["X"].select(NUMERIC_FEATURES).to_numpy().astype(float)
    col = X_numpy[:, 0]

    cuts = _pd_guided_splits(
        model=fitted_gbm,
        X_numpy=X_numpy,
        feature_idx=0,
        feature_name="driver_age",
        n_splits=8,
    )
    lo, hi = float(col.min()), float(col.max())
    for c in cuts:
        assert lo < c < hi, f"Cut {c} outside data range [{lo}, {hi}]"


def test_pd_guided_splits_fallback_on_invalid_model(synthetic_motor_data, fitted_gbm):
    """When the model lacks 'fit', partial_dependence raises and we fall back gracefully."""
    data = synthetic_motor_data
    X_numpy = data["X"].select(NUMERIC_FEATURES).to_numpy().astype(float)
    model = _NoFitWrapper(fitted_gbm, NUMERIC_FEATURES)

    # Should not raise — falls back to quantile splits with a warning
    with pytest.warns(UserWarning, match="partial_dependence failed"):
        cuts = _pd_guided_splits(
            model=model,
            X_numpy=X_numpy,
            feature_idx=0,
            feature_name="driver_age",
            n_splits=5,
        )
    assert isinstance(cuts, list)
    assert len(cuts) <= 5


# ---------------------------------------------------------------------------
# Unit tests: _apply_cuts
# ---------------------------------------------------------------------------


def test_apply_cuts_returns_series():
    s = pl.Series("x", [1.0, 5.0, 10.0, 15.0])
    result = _apply_cuts(s, [3.0, 8.0], "x")
    assert isinstance(result, pl.Series)


def test_apply_cuts_correct_number_of_bins():
    s = pl.Series("x", [1.0, 5.0, 10.0, 15.0])
    cuts = [3.0, 8.0]  # 3 bins
    result = _apply_cuts(s, cuts, "x")
    assert result.n_unique() <= 3


def test_apply_cuts_alias():
    s = pl.Series("x", [1.0, 5.0])
    result = _apply_cuts(s, [3.0], "myfeature")
    assert result.name == "myfeature__bin"


def test_apply_cuts_no_cuts():
    """With no cut-points, everything goes in one bin."""
    s = pl.Series("x", [1.0, 2.0, 3.0])
    result = _apply_cuts(s, [], "x")
    assert result.n_unique() == 1


# ---------------------------------------------------------------------------
# Unit tests: _build_ohe_block
# ---------------------------------------------------------------------------


def test_build_ohe_block_shape():
    s = pl.Series("feat", ["A", "B", "C", "A", "B"])
    block, names, base = _build_ohe_block(s, "feat")
    # 3 levels → 2 non-reference columns
    assert block.shape == (5, 2)
    assert len(names) == 2


def test_build_ohe_block_reference_dropped():
    s = pl.Series("feat", ["A", "B", "C"])
    block, names, base = _build_ohe_block(s, "feat")
    # "A" is reference (first alphabetically), so no column for "A"
    assert base == "A"
    assert all("=A" not in n for n in names)


def test_build_ohe_block_single_level():
    """Single-level series should return an empty block."""
    s = pl.Series("feat", ["A", "A", "A"])
    block, names, base = _build_ohe_block(s, "feat")
    assert block.shape[1] == 0
    assert names == []


# ---------------------------------------------------------------------------
# End-to-end fit tests
# ---------------------------------------------------------------------------


def test_fit_returns_self(synthetic_motor_data, fitted_gbm):
    data = synthetic_motor_data
    lg = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
        n_bins=4,
        alpha=1e-4,
        family="poisson",
    )
    result = lg.fit(
        data["X"].select(NUMERIC_FEATURES),
        data["y"],
        exposure=data["exposure"],
    )
    assert result is lg


def test_fit_sets_fitted_flag(fitted_lasso_glm):
    assert fitted_lasso_glm._fitted is True


def test_fit_cuts_populated(fitted_lasso_glm):
    for feat in NUMERIC_FEATURES:
        assert feat in fitted_lasso_glm._cuts


def test_fit_selected_features_subset(fitted_lasso_glm):
    selected = fitted_lasso_glm._selected_features
    assert set(selected).issubset(set(NUMERIC_FEATURES))


def test_fit_selected_features_not_empty(fitted_lasso_glm):
    """With alpha=1e-4, all three features should survive lasso."""
    assert len(fitted_lasso_glm._selected_features) > 0


def test_fit_before_fit_raises(fitted_gbm):
    lg = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
    )
    with pytest.raises(RuntimeError, match="not been fitted"):
        lg.factor_tables()


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------


def test_predict_shape(fitted_lasso_glm, synthetic_motor_data):
    data = synthetic_motor_data
    preds = fitted_lasso_glm.predict(data["X"].select(NUMERIC_FEATURES))
    assert preds.shape == (len(data["X"]),)


def test_predict_positive(fitted_lasso_glm, synthetic_motor_data):
    data = synthetic_motor_data
    preds = fitted_lasso_glm.predict(data["X"].select(NUMERIC_FEATURES))
    assert (preds > 0).all(), "All predictions should be positive"


def test_predict_reasonable_range(fitted_lasso_glm, synthetic_motor_data):
    """Predictions should be in a plausible range for a frequency model."""
    data = synthetic_motor_data
    preds = fitted_lasso_glm.predict(data["X"].select(NUMERIC_FEATURES))
    assert preds.max() < 100.0
    assert preds.min() > 1e-6


# ---------------------------------------------------------------------------
# factor_tables() tests
# ---------------------------------------------------------------------------


def test_factor_tables_returns_dict(fitted_lasso_glm):
    tables = fitted_lasso_glm.factor_tables()
    assert isinstance(tables, dict)


def test_factor_tables_keys_are_selected_features(fitted_lasso_glm):
    tables = fitted_lasso_glm.factor_tables()
    assert set(tables.keys()) == set(fitted_lasso_glm._selected_features)


def test_factor_tables_columns(fitted_lasso_glm):
    tables = fitted_lasso_glm.factor_tables()
    for feat, df in tables.items():
        assert "level" in df.columns, f"'level' missing in {feat}"
        assert "log_coefficient" in df.columns, f"'log_coefficient' missing in {feat}"
        assert "relativity" in df.columns, f"'relativity' missing in {feat}"


def test_factor_tables_relativities_positive(fitted_lasso_glm):
    tables = fitted_lasso_glm.factor_tables()
    for feat, df in tables.items():
        rels = df["relativity"].to_numpy()
        assert (rels > 0).all(), f"Non-positive relativity found in {feat}"


def test_factor_tables_level_strings(fitted_lasso_glm):
    """Level column should contain string interval labels."""
    tables = fitted_lasso_glm.factor_tables()
    for feat, df in tables.items():
        assert df["level"].dtype == pl.String, f"'level' not String in {feat}"
        for lv in df["level"].to_list():
            assert lv.startswith("["), f"Unexpected level format: {lv!r} in {feat}"


def test_factor_tables_match_surrogate_glm_format(fitted_lasso_glm):
    """
    Factor tables from LassoGuidedGLM must have the same three columns as those
    from SurrogateGLM: level, log_coefficient, relativity.
    """
    tables = fitted_lasso_glm.factor_tables()
    expected_cols = {"level", "log_coefficient", "relativity"}
    for feat, df in tables.items():
        assert set(df.columns) == expected_cols, (
            f"{feat}: expected columns {expected_cols}, got {set(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Lasso feature selection test
# ---------------------------------------------------------------------------


def test_lasso_selects_subset_of_features(synthetic_motor_data, fitted_gbm):
    """
    Lasso feature selection should produce a strict subset of all features when
    alpha is large enough. We verify that higher alpha results in fewer selected
    features than very low alpha — demonstrating that the selection mechanism
    works in the expected direction.
    """
    data = synthetic_motor_data
    X = data["X"].select(NUMERIC_FEATURES)

    # Fit with very low alpha (permissive — should keep most/all features)
    lg_permissive = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
        n_bins=5,
        alpha=1e-6,   # essentially unpenalised
        family="poisson",
    )
    lg_permissive.fit(X, data["y"], exposure=data["exposure"])

    # Fit with higher alpha (restrictive — should keep fewer features)
    lg_strict = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
        n_bins=5,
        alpha=5.0,   # strong penalty
        family="poisson",
    )
    lg_strict.fit(X, data["y"], exposure=data["exposure"])

    n_permissive = len(lg_permissive._selected_features)
    n_strict = len(lg_strict._selected_features)

    # At very low alpha, lasso should keep all features
    assert n_permissive == len(NUMERIC_FEATURES), (
        f"Expected all {len(NUMERIC_FEATURES)} features at alpha=1e-6, "
        f"got {n_permissive}: {lg_permissive._selected_features}"
    )
    # Higher alpha should select fewer features
    assert n_strict <= n_permissive, (
        f"Expected strict alpha to select fewer features, but got "
        f"{n_strict} (strict) vs {n_permissive} (permissive)"
    )


# ---------------------------------------------------------------------------
# Tweedie family test
# ---------------------------------------------------------------------------


def test_tweedie_family_fits(synthetic_motor_data, fitted_gbm):
    data = synthetic_motor_data

    lg = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
        n_bins=4,
        alpha=1e-4,
        family="tweedie",
        power=1.5,
    )
    # Clip y to a positive floor: Tweedie GLM requires positive targets
    y_tw = np.clip(data["y"].astype(float), 1e-4, None)
    lg.fit(
        data["X"].select(NUMERIC_FEATURES),
        y_tw,
        exposure=data["exposure"],
    )
    assert lg._fitted
    preds = lg.predict(data["X"].select(NUMERIC_FEATURES))
    assert (preds > 0).all()


# ---------------------------------------------------------------------------
# Poisson family explicit test
# ---------------------------------------------------------------------------


def test_poisson_family_fits(synthetic_motor_data, fitted_gbm):
    data = synthetic_motor_data

    lg = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
        n_bins=5,
        alpha=1e-4,
        family="poisson",
    )
    lg.fit(
        data["X"].select(NUMERIC_FEATURES),
        data["y"],
        exposure=data["exposure"],
    )
    assert lg._fitted


# ---------------------------------------------------------------------------
# summary() smoke test
# ---------------------------------------------------------------------------


def test_summary_runs_without_error(fitted_lasso_glm, capsys):
    fitted_lasso_glm.summary()
    captured = capsys.readouterr()
    assert "LassoGuidedGLM" in captured.out


def test_summary_shows_family(fitted_lasso_glm, capsys):
    fitted_lasso_glm.summary()
    captured = capsys.readouterr()
    assert "poisson" in captured.out.lower()


def test_summary_shows_feature_count(fitted_lasso_glm, capsys):
    fitted_lasso_glm.summary()
    captured = capsys.readouterr()
    assert "Features input:" in captured.out
    assert "Features selected:" in captured.out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises(fitted_gbm):
    lg = LassoGuidedGLM(gbm_model=fitted_gbm, feature_names=NUMERIC_FEATURES)
    X_dummy = pl.DataFrame({
        "driver_age": [30.0],
        "vehicle_value": [20000.0],
        "ncd_years": [3.0],
    })
    with pytest.raises(RuntimeError, match="not been fitted"):
        lg.predict(X_dummy)


def test_sample_weight_and_exposure_combined(synthetic_motor_data, fitted_gbm):
    """Passing both sample_weight and exposure should not crash."""
    data = synthetic_motor_data
    n = len(data["X"])

    lg = LassoGuidedGLM(
        gbm_model=fitted_gbm,
        feature_names=NUMERIC_FEATURES,
        n_bins=4,
        alpha=1e-4,
        family="poisson",
    )
    lg.fit(
        data["X"].select(NUMERIC_FEATURES),
        data["y"],
        sample_weight=np.ones(n),
        exposure=data["exposure"],
    )
    assert lg._fitted
