"""
Expanded test coverage for insurance-distill.

Covers gaps in:
  - _types.py: BinSpec.apply(), ValidationMetrics.summary(), DistillationReport.__repr__
  - _export.py: build_factor_tables() with categorical features, format_radar_csv edge cases
  - _surrogate.py: interaction_pairs, categorical_features, tweedie family, predict_method param,
                   unknown family error, tweedie family fit, quantile binning method
  - build_factor_tables: base level placeholder logic
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from insurance_distill import (
    SurrogateGLM,
    BinSpec,
    ValidationMetrics,
    DistillationReport,
    build_factor_tables,
    build_glm_coefficients_df,
    format_radar_csv,
)

NUMERIC_FEATURES = ["driver_age", "vehicle_value", "ncd_years"]


# ---------------------------------------------------------------------------
# Fixtures: reuse conftest fixtures + add a categorical-aware wrapper
# ---------------------------------------------------------------------------


class NumericOnlyPredictWrapper:
    """Accepts full DataFrame with extra columns; ignores non-numeric."""

    def __init__(self, gbm, numeric_cols: list[str]) -> None:
        self._gbm = gbm
        self._cols = numeric_cols

    def predict(self, X):
        if isinstance(X, pl.DataFrame):
            return self._gbm.predict(X.select(self._cols).to_numpy())
        return self._gbm.predict(X)


# ---------------------------------------------------------------------------
# BinSpec.apply() tests
# ---------------------------------------------------------------------------


class TestBinSpecApply:
    def _make_spec(self) -> BinSpec:
        return BinSpec(
            feature="age",
            bin_edges=[-float("inf"), 25.0, 50.0, float("inf")],
            bin_labels=["[−inf, 25.00)", "[25.00, 50.00)", "[50.00, +inf)"],
            method="tree",
            n_bins=3,
        )

    def test_apply_returns_series(self):
        spec = self._make_spec()
        s = pl.Series("age", [20.0, 30.0, 60.0])
        result = spec.apply(s)
        assert isinstance(result, pl.Series)

    def test_apply_length_unchanged(self):
        spec = self._make_spec()
        s = pl.Series("age", [20.0, 30.0, 60.0, 40.0, 18.0])
        result = spec.apply(s)
        assert len(result) == 5

    def test_apply_produces_correct_bins(self):
        spec = self._make_spec()
        s = pl.Series("age", [20.0, 30.0, 60.0])
        result = spec.apply(s)
        vals = result.cast(pl.String).to_list()
        # 20 -> first bin, 30 -> second bin, 60 -> third bin
        assert vals[0] != vals[1]  # 20 and 30 in different bins
        assert vals[1] != vals[2]  # 30 and 60 in different bins

    def test_apply_no_nulls_for_in_range_values(self):
        spec = self._make_spec()
        s = pl.Series("age", [19.0, 25.5, 51.0])
        result = spec.apply(s)
        assert result.null_count() == 0

    def test_apply_single_bin(self):
        spec = BinSpec(
            feature="x",
            bin_edges=[-float("inf"), float("inf")],
            bin_labels=["all"],
            method="quantile",
            n_bins=1,
        )
        s = pl.Series("x", [1.0, 2.0, 3.0])
        result = spec.apply(s)
        assert result.n_unique() == 1


# ---------------------------------------------------------------------------
# ValidationMetrics.summary() tests
# ---------------------------------------------------------------------------


class TestValidationMetricsSummary:
    def _make_metrics(self, **overrides) -> ValidationMetrics:
        defaults = dict(
            gini_gbm=0.45,
            gini_glm=0.40,
            gini_ratio=0.40 / 0.45,
            deviance_ratio=0.35,
            max_segment_deviation=0.12,
            mean_segment_deviation=0.04,
            n_segments=120,
        )
        defaults.update(overrides)
        return ValidationMetrics(**defaults)

    def test_summary_returns_string(self):
        m = self._make_metrics()
        assert isinstance(m.summary(), str)

    def test_summary_contains_gini(self):
        m = self._make_metrics()
        assert "Gini" in m.summary()

    def test_summary_contains_percent(self):
        m = self._make_metrics()
        assert "%" in m.summary()

    def test_summary_contains_deviance_ratio(self):
        m = self._make_metrics()
        assert "Deviance" in m.summary()

    def test_summary_contains_segment_count(self):
        m = self._make_metrics(n_segments=999)
        assert "999" in m.summary()

    def test_summary_contains_max_segment_deviation(self):
        m = self._make_metrics(max_segment_deviation=0.25)
        assert "25" in m.summary()

    def test_summary_contains_mean_segment_deviation(self):
        m = self._make_metrics(mean_segment_deviation=0.08)
        # 0.08 formatted as 8.0% or similar
        s = m.summary()
        assert "Mean segment" in s or "mean segment" in s.lower()


# ---------------------------------------------------------------------------
# DistillationReport.__repr__ tests
# ---------------------------------------------------------------------------


class TestDistillationReportRepr:
    def _make_report(self) -> DistillationReport:
        metrics = ValidationMetrics(
            gini_gbm=0.4,
            gini_glm=0.36,
            gini_ratio=0.9,
            deviance_ratio=0.3,
            max_segment_deviation=0.05,
            mean_segment_deviation=0.02,
            n_segments=50,
        )
        return DistillationReport(
            metrics=metrics,
            factor_tables={"age": pl.DataFrame({"level": ["A"], "relativity": [1.0], "log_coefficient": [0.0]})},
        )

    def test_repr_contains_class_name(self):
        report = self._make_report()
        r = repr(report)
        assert "DistillationReport" in r

    def test_repr_contains_gini_ratio(self):
        report = self._make_report()
        r = repr(report)
        assert "gini_ratio" in r

    def test_repr_contains_max_segment_deviation(self):
        report = self._make_report()
        r = repr(report)
        assert "max_segment_deviation" in r

    def test_repr_contains_feature_list(self):
        report = self._make_report()
        r = repr(report)
        assert "age" in r


# ---------------------------------------------------------------------------
# build_factor_tables: categorical features
# ---------------------------------------------------------------------------


class TestBuildFactorTablesCategorical:
    class _FakeGLM:
        coef_ = np.array([0.3, -0.1, 0.5])
        intercept_ = -2.5

    def test_categorical_feature_in_output(self):
        """build_factor_tables should include categorical features in tables."""
        tables = build_factor_tables(
            glm=self._FakeGLM(),
            bin_specs={},  # no continuous features
            col_names=["region=South", "region=East", "region=West"],
            intercept=-2.5,
            cat_features=["region"],
        )
        assert "region" in tables

    def test_categorical_table_has_base_row(self):
        tables = build_factor_tables(
            glm=self._FakeGLM(),
            bin_specs={},
            col_names=["region=South", "region=East", "region=West"],
            intercept=-2.5,
            cat_features=["region"],
        )
        region_df = tables["region"]
        levels = region_df["level"].to_list()
        assert any("base" in lv.lower() or "reference" in lv.lower() for lv in levels)

    def test_categorical_table_relativities_positive(self):
        tables = build_factor_tables(
            glm=self._FakeGLM(),
            bin_specs={},
            col_names=["channel=broker", "channel=online"],
            intercept=-2.0,
            cat_features=["channel"],
        )
        rels = tables["channel"]["relativity"].to_numpy()
        assert (rels > 0).all()

    def test_categorical_table_base_relativity_is_one(self):
        tables = build_factor_tables(
            glm=self._FakeGLM(),
            bin_specs={},
            col_names=["region=South", "region=East", "region=West"],
            intercept=-2.5,
            cat_features=["region"],
        )
        region_df = tables["region"]
        base_row = region_df.filter(pl.col("log_coefficient") == 0.0)
        assert len(base_row) >= 1
        assert float(base_row["relativity"][0]) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# format_radar_csv edge cases
# ---------------------------------------------------------------------------


class TestFormatRadarCsv:
    def _make_table(self) -> pl.DataFrame:
        return pl.DataFrame({
            "level": ["[-inf, 25.00)", "[25.00, 50.00)", "[50.00, +inf)"],
            "log_coefficient": [0.1, 0.0, -0.2],
            "relativity": [1.105, 1.000, 0.819],
        })

    def test_header_row(self):
        csv_str = format_radar_csv(self._make_table(), "driver_age")
        header = csv_str.split("\n")[0]
        assert header == "driver_age,Relativity"

    def test_correct_number_of_rows(self):
        csv_str = format_radar_csv(self._make_table(), "driver_age")
        lines = [l for l in csv_str.strip().split("\n") if l]
        assert len(lines) == 4  # header + 3 bins

    def test_six_decimal_precision(self):
        csv_str = format_radar_csv(self._make_table(), "driver_age")
        lines = csv_str.strip().split("\n")
        # Every data line should have exactly 6 decimal places
        for line in lines[1:]:
            rel_part = line.split(",")[1]
            decimal_part = rel_part.split(".")[1]
            assert len(decimal_part) == 6, f"Expected 6 decimal places, got: {rel_part!r}"

    def test_feature_name_in_header(self):
        csv_str = format_radar_csv(self._make_table(), "vehicle_value")
        assert csv_str.startswith("vehicle_value,")

    def test_single_bin_table(self):
        df = pl.DataFrame({
            "level": ["all"],
            "log_coefficient": [0.0],
            "relativity": [1.0],
        })
        csv_str = format_radar_csv(df, "x")
        lines = [l for l in csv_str.strip().split("\n") if l]
        assert len(lines) == 2  # header + 1 bin

    def test_ends_with_newline(self):
        csv_str = format_radar_csv(self._make_table(), "x")
        assert csv_str.endswith("\n")


# ---------------------------------------------------------------------------
# SurrogateGLM: interaction pairs
# ---------------------------------------------------------------------------


class TestSurrogateGLMInteractions:
    def test_interaction_pairs_fit_runs(self, synthetic_motor_data, fitted_gbm):
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
        )
        # driver_age and ncd_years are genuinely correlated in motor insurance
        surrogate.fit(
            features=NUMERIC_FEATURES,
            max_bins=4,
            interaction_pairs=[("driver_age", "ncd_years")],
        )
        assert surrogate._fitted

    def test_interaction_pairs_expand_design_columns(self, synthetic_motor_data, fitted_gbm):
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)

        surrogate_plain = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
        )
        surrogate_plain.fit(features=NUMERIC_FEATURES, max_bins=3)

        surrogate_inter = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
        )
        surrogate_inter.fit(
            features=NUMERIC_FEATURES,
            max_bins=3,
            interaction_pairs=[("driver_age", "ncd_years")],
        )

        # Interaction model should have more design columns
        n_plain = len(surrogate_plain._design_col_names)
        n_inter = len(surrogate_inter._design_col_names)
        assert n_inter > n_plain, (
            f"Expected more columns with interactions: {n_inter} > {n_plain}"
        )

    def test_invalid_interaction_pair_warns(self, synthetic_motor_data, fitted_gbm):
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
        )
        # "nonexistent" column should produce a warning but not crash
        with pytest.warns(UserWarning, match="skipped"):
            surrogate.fit(
                features=NUMERIC_FEATURES,
                max_bins=3,
                interaction_pairs=[("driver_age", "nonexistent")],
            )
        assert surrogate._fitted


# ---------------------------------------------------------------------------
# SurrogateGLM: categorical features
# ---------------------------------------------------------------------------


class TestSurrogateGLMCategoricalFeatures:
    def test_categorical_features_included_in_factor_tables(
        self, synthetic_motor_data, fitted_gbm_with_region
    ):
        """Region should appear in factor tables when passed as categorical."""
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm_with_region, NUMERIC_FEATURES)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
        )
        surrogate.fit(
            features=NUMERIC_FEATURES,
            categorical_features=["region"],
            max_bins=4,
        )
        report = surrogate.report()
        assert "region" in report.factor_tables, (
            f"Expected 'region' in factor tables. Got: {list(report.factor_tables.keys())}"
        )

    def test_categorical_feature_relativities_positive(
        self, synthetic_motor_data, fitted_gbm_with_region
    ):
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm_with_region, NUMERIC_FEATURES)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
        )
        surrogate.fit(
            features=NUMERIC_FEATURES,
            categorical_features=["region"],
            max_bins=4,
        )
        report = surrogate.report()
        rels = report.factor_tables["region"]["relativity"].to_numpy()
        assert (rels > 0).all()


# ---------------------------------------------------------------------------
# SurrogateGLM: tweedie family
# ---------------------------------------------------------------------------


class TestSurrogateGLMTweedie:
    def test_tweedie_fit_succeeds(self, synthetic_motor_data, fitted_gbm):
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)

        # Tweedie requires positive target values
        y_pos = np.clip(data["y"].astype(float), 0.01, None)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=y_pos,
            exposure=data["exposure"],
            family="tweedie",
        )
        surrogate.fit(features=NUMERIC_FEATURES, max_bins=4)
        assert surrogate._fitted

    def test_tweedie_report_returns_distillation_report(self, synthetic_motor_data, fitted_gbm):
        from insurance_distill import DistillationReport

        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)
        y_pos = np.clip(data["y"].astype(float), 0.01, None)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=y_pos,
            exposure=data["exposure"],
            family="tweedie",
        )
        surrogate.fit(features=NUMERIC_FEATURES, max_bins=4)
        report = surrogate.report()
        assert isinstance(report, DistillationReport)


# ---------------------------------------------------------------------------
# SurrogateGLM: unknown family raises ValueError
# ---------------------------------------------------------------------------


class TestSurrogateGLMUnknownFamily:
    def test_unknown_family_raises_on_fit(self, synthetic_motor_data, fitted_gbm):
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="negbinomial",  # type: ignore — deliberately invalid
        )
        with pytest.raises((ValueError, Exception)):
            surrogate.fit(features=NUMERIC_FEATURES, max_bins=4)


# ---------------------------------------------------------------------------
# SurrogateGLM: quantile binning method
# ---------------------------------------------------------------------------


class TestSurrogateGLMQuantileBinning:
    def test_quantile_binning_fits(self, synthetic_motor_data, fitted_gbm):
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
        )
        surrogate.fit(
            features=NUMERIC_FEATURES,
            max_bins=5,
            binning_method="quantile",
        )
        assert surrogate._fitted
        # Quantile bins should have method="quantile" recorded
        for feat in NUMERIC_FEATURES:
            assert surrogate._bin_specs[feat].method == "quantile"


# ---------------------------------------------------------------------------
# SurrogateGLM: predict_method parameter
# ---------------------------------------------------------------------------


class TestSurrogateGLMPredictMethod:
    def test_explicit_predict_method(self, synthetic_motor_data, fitted_gbm):
        """Explicit predict_method='predict' should work for sklearn models."""
        data = synthetic_motor_data
        wrapper = NumericOnlyPredictWrapper(fitted_gbm, NUMERIC_FEATURES)

        surrogate = SurrogateGLM(
            model=wrapper,
            X_train=data["X"],
            y_train=data["y"],
            exposure=data["exposure"],
            family="poisson",
            predict_method="predict",
        )
        surrogate.fit(features=NUMERIC_FEATURES, max_bins=4)
        assert surrogate._fitted
        assert (surrogate._pseudo_predictions > 0).all()

    def test_missing_predict_method_raises(self, synthetic_motor_data):
        """A model with no predict/predict_proba should raise AttributeError."""
        data = synthetic_motor_data

        class _NoPredict:
            pass

        surrogate = SurrogateGLM(
            model=_NoPredict(),
            X_train=data["X"],
            y_train=data["y"],
            family="poisson",
        )
        with pytest.raises(AttributeError, match="predict"):
            surrogate.fit(features=NUMERIC_FEATURES, max_bins=3)


# ---------------------------------------------------------------------------
# build_glm_coefficients_df: edge cases
# ---------------------------------------------------------------------------


class TestBuildGLMCoeffDfEdgeCases:
    class _FakeGLM:
        coef_ = np.array([])
        intercept_ = -1.5

    def test_intercept_only_model(self):
        """Model with no feature coefficients should produce just the intercept row."""
        df = build_glm_coefficients_df(
            glm=self._FakeGLM(),
            col_names=[],
            intercept=-1.5,
        )
        assert len(df) == 1
        assert df["term"][0] == "(Intercept)"

    def test_relativity_equals_exp_log_coefficient(self):
        class _FakeGLM2:
            coef_ = np.array([0.2, -0.4])
            intercept_ = -2.0

        df = build_glm_coefficients_df(
            glm=_FakeGLM2(),
            col_names=["age=25", "age=30"],
            intercept=-2.0,
        )
        lc = df["log_coefficient"].to_numpy()
        rel = df["relativity"].to_numpy()
        assert np.allclose(rel, np.exp(lc), rtol=1e-10)
