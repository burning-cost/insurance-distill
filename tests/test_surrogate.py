"""Tests for _surrogate.py - SurrogateGLM end-to-end."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_distill import SurrogateGLM


NUMERIC_FEATURES = ["driver_age", "vehicle_value", "ncd_years"]


@pytest.fixture
def simple_surrogate(synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
    """A fitted SurrogateGLM on numeric features only."""
    data = synthetic_motor_data
    model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)

    surrogate = SurrogateGLM(
        model=model,
        X_train=data["X"],
        y_train=data["y"],
        exposure=data["exposure"],
        family="poisson",
    )
    surrogate.fit(features=NUMERIC_FEATURES, max_bins=5)
    return surrogate


# ---------------------------------------------------------------------------
# Fit smoke tests
# ---------------------------------------------------------------------------


def test_fit_returns_self(synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
    data = synthetic_motor_data
    model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
    surrogate = SurrogateGLM(
        model=model,
        X_train=data["X"],
        y_train=data["y"],
        exposure=data["exposure"],
        family="poisson",
    )
    result = surrogate.fit(features=NUMERIC_FEATURES, max_bins=5)
    assert result is surrogate


def test_fit_populates_bin_specs(simple_surrogate):
    assert len(simple_surrogate._bin_specs) == 3
    assert "driver_age" in simple_surrogate._bin_specs


def test_fit_pseudo_predictions_positive(simple_surrogate):
    assert (simple_surrogate._pseudo_predictions > 0).all()


def test_fit_glm_predictions_positive(simple_surrogate):
    assert (simple_surrogate._glm_predictions > 0).all()


def test_fit_before_fit_raises(synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
    data = synthetic_motor_data
    model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
    surrogate = SurrogateGLM(
        model=model,
        X_train=data["X"],
        y_train=data["y"],
        family="poisson",
    )
    with pytest.raises(RuntimeError, match="not been fitted"):
        surrogate.report()


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


def test_report_returns_distillation_report(simple_surrogate):
    from insurance_distill import DistillationReport

    report = simple_surrogate.report()
    assert isinstance(report, DistillationReport)


def test_report_metrics_gini_ratio_reasonable(simple_surrogate):
    """Gini ratio should be between 0 and 1."""
    report = simple_surrogate.report()
    gini_ratio = report.metrics.gini_ratio
    assert 0.0 <= gini_ratio <= 1.0


def test_report_metrics_deviance_ratio_nonnegative(simple_surrogate):
    report = simple_surrogate.report()
    assert report.metrics.deviance_ratio >= 0.0


def test_report_factor_tables_present(simple_surrogate):
    report = simple_surrogate.report()
    assert set(report.factor_tables.keys()) == set(NUMERIC_FEATURES)


def test_report_factor_table_columns(simple_surrogate):
    report = simple_surrogate.report()
    for feat, df in report.factor_tables.items():
        assert "level" in df.columns
        assert "relativity" in df.columns
        assert "log_coefficient" in df.columns


def test_report_factor_table_relativities_positive(simple_surrogate):
    report = simple_surrogate.report()
    for feat, df in report.factor_tables.items():
        rels = df["relativity"].to_numpy()
        assert (rels > 0).all(), f"Non-positive relativity in {feat}"


def test_report_glm_coefficients_shape(simple_surrogate):
    report = simple_surrogate.report()
    coeff_df = report.glm_coefficients
    assert coeff_df is not None
    assert "term" in coeff_df.columns
    assert "relativity" in coeff_df.columns
    # Should include intercept + all non-reference levels
    assert len(coeff_df) >= 2


def test_report_lift_chart_present(simple_surrogate):
    report = simple_surrogate.report()
    assert report.lift_chart is not None
    assert "decile" in report.lift_chart.columns


def test_report_summary_string(simple_surrogate):
    report = simple_surrogate.report()
    summary = report.metrics.summary()
    assert "Gini" in summary
    assert "%" in summary


# ---------------------------------------------------------------------------
# factor_table() method
# ---------------------------------------------------------------------------


def test_factor_table_driver_age(simple_surrogate):
    df = simple_surrogate.factor_table("driver_age")
    assert len(df) > 0
    assert df["relativity"].min() > 0


def test_factor_table_unknown_feature_raises(simple_surrogate):
    with pytest.raises(KeyError, match="nonexistent"):
        simple_surrogate.factor_table("nonexistent")


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


def test_export_csv_creates_files(simple_surrogate, tmp_path):
    written = simple_surrogate.export_csv(str(tmp_path))
    assert len(written) > 0
    for path in written:
        import os
        assert os.path.exists(path)


def test_export_csv_base_file(simple_surrogate, tmp_path):
    simple_surrogate.export_csv(str(tmp_path), include_base=True)
    import os
    assert os.path.exists(os.path.join(str(tmp_path), "base.csv"))


def test_export_csv_readable_by_polars(simple_surrogate, tmp_path):
    simple_surrogate.export_csv(str(tmp_path))
    import os
    for fname in os.listdir(str(tmp_path)):
        if fname.endswith(".csv"):
            df = pl.read_csv(os.path.join(str(tmp_path), fname))
            assert len(df) > 0


# ---------------------------------------------------------------------------
# Gamma family
# ---------------------------------------------------------------------------


def test_gamma_family_fits(synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
    data = synthetic_motor_data
    model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
    surrogate = SurrogateGLM(
        model=model,
        X_train=data["X"],
        y_train=data["y"],
        exposure=data["exposure"],
        family="gamma",
    )
    surrogate.fit(features=NUMERIC_FEATURES, max_bins=4)
    report = surrogate.report()
    assert report.metrics.gini_ratio >= 0.0


# ---------------------------------------------------------------------------
# No exposure (unit exposure)
# ---------------------------------------------------------------------------


def test_no_exposure_works(synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
    data = synthetic_motor_data
    model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
    surrogate = SurrogateGLM(
        model=model,
        X_train=data["X"],
        y_train=data["y"],
        exposure=None,  # unit exposure
        family="poisson",
    )
    surrogate.fit(features=NUMERIC_FEATURES, max_bins=4)
    report = surrogate.report()
    assert report.metrics.gini_ratio >= 0.0


# ---------------------------------------------------------------------------
# Binning method overrides
# ---------------------------------------------------------------------------


def test_method_override_isotonic(synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
    data = synthetic_motor_data
    model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
    surrogate = SurrogateGLM(
        model=model,
        X_train=data["X"],
        y_train=data["y"],
        exposure=data["exposure"],
        family="poisson",
    )
    surrogate.fit(
        features=NUMERIC_FEATURES,
        max_bins=6,
        binning_method="tree",
        method_overrides={"ncd_years": "isotonic"},
    )
    assert surrogate._bin_specs["ncd_years"].method == "isotonic"


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_distillation_report_repr(simple_surrogate):
    report = simple_surrogate.report()
    r = repr(report)
    assert "DistillationReport" in r
    assert "gini_ratio" in r


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------


class TestSurrogateGLMConstructorValidation:
    """SurrogateGLM should raise ValueError at construction for invalid params."""

    def test_alpha_negative_raises(self, synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
        data = synthetic_motor_data
        model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
        with pytest.raises(ValueError, match="alpha"):
            SurrogateGLM(
                model=model,
                X_train=data["X"],
                y_train=data["y"],
                exposure=data["exposure"],
                family="poisson",
                alpha=-1.0,
            )

    def test_alpha_zero_is_valid(self, synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
        """alpha=0.0 is the default (unregularised) — must not raise."""
        data = synthetic_motor_data
        model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
        surrogate = SurrogateGLM(
            model=model,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
            alpha=0.0,
        )
        assert surrogate.alpha == 0.0

    def test_positive_alpha_is_valid(self, synthetic_motor_data, fitted_gbm, numeric_only_wrapper_class):
        data = synthetic_motor_data
        model = numeric_only_wrapper_class(fitted_gbm, NUMERIC_FEATURES)
        surrogate = SurrogateGLM(
            model=model,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
            alpha=0.5,
        )
        assert surrogate.alpha == 0.5
