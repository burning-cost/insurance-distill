"""
insurance-distill: Distil GBM models into multiplicative GLM factor tables.

The problem this library solves:
    You have a CatBoost model that outperforms your GLM in validation, but
    your rating engine (Radar, Emblem, or any multiplicative system) needs
    factor tables. This library bridges that gap by fitting a GLM surrogate
    on the GBM's predictions, then exporting the result as factor tables.

Quick start::

    from insurance_distill import SurrogateGLM

    surrogate = SurrogateGLM(
        model=fitted_catboost,
        X_train=X_train,
        y_train=y_train,
        exposure=exposure_arr,
        family="poisson",
    )
    surrogate.fit(max_bins=10)

    report = surrogate.report()
    print(report.metrics.summary())

    surrogate.export_csv("output/factors/")
"""
from ._surrogate import SurrogateGLM
from ._binning import OptimalBinner
from ._types import BinSpec, ValidationMetrics, DistillationReport
from ._validation import (
    compute_gini,
    compute_deviance_ratio,
    compute_segment_deviation,
    double_lift_chart,
)
from ._export import build_factor_tables, build_glm_coefficients_df, format_radar_csv

__all__ = [
    "SurrogateGLM",
    "OptimalBinner",
    "BinSpec",
    "ValidationMetrics",
    "DistillationReport",
    "compute_gini",
    "compute_deviance_ratio",
    "compute_segment_deviation",
    "double_lift_chart",
    "build_factor_tables",
    "build_glm_coefficients_df",
    "format_radar_csv",
]

__version__ = "0.1.0"
