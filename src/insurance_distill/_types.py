"""
Type definitions and result dataclasses for insurance-distill.

We keep types in their own module so that the other modules can import
from here without circular dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class BinSpec:
    """
    Describes the binning applied to one variable.

    Attributes
    ----------
    feature : str
        Column name in the training DataFrame.
    bin_edges : list[float]
        Cut-points that define the bins, including -inf and +inf as sentinels
        at the boundaries. Length == n_bins + 1.
    bin_labels : list[str]
        Human-readable labels for each bin (used in factor tables).
    method : str
        Binning method used: ``"tree"``, ``"quantile"``, or ``"isotonic"``.
    n_bins : int
        Number of bins actually produced (may be less than ``max_bins`` if the
        variable does not have enough distinct values).
    """

    feature: str
    bin_edges: list[float]
    bin_labels: list[str]
    method: str
    n_bins: int

    def apply(self, series: pl.Series) -> pl.Series:
        """
        Map a Polars Series of floats to bin labels (string category).

        Rows outside all bins (should not happen in practice) are labelled
        ``"Unknown"``.
        """
        edges = self.bin_edges  # includes -inf, cut1, cut2, ..., +inf
        labels = self.bin_labels

        # polars cut: edges are the interior breaks (excluding -inf / +inf)
        interior = [e for e in edges if not (np.isinf(e))]
        result = series.cast(pl.Float64).cut(interior, labels=labels, left_closed=True)
        return result


@dataclass
class ValidationMetrics:
    """
    Scalar validation metrics produced by :class:`~insurance_distill.SurrogateGLM`.

    Attributes
    ----------
    gini_gbm : float
        Gini coefficient of the original GBM pseudo-predictions on the
        training sample, used as the reference.
    gini_glm : float
        Gini coefficient of the fitted surrogate GLM on the same sample.
    gini_ratio : float
        ``gini_glm / gini_gbm``. Values above 0.90 are generally acceptable
        for most rating engines; above 0.95 is excellent.
    deviance_ratio : float
        ``1 - null_deviance_glm / residual_deviance_glm``. Analogous to
        R-squared for GLMs. Higher is better.
    max_segment_deviation : float
        Maximum absolute relative deviation between the GBM and GLM fitted
        values, aggregated across the finest available segments (i.e. each
        unique combination of binned features).
    mean_segment_deviation : float
        Mean absolute relative deviation across segments (exposure-weighted).
    n_segments : int
        Number of distinct segments used to compute segment deviation.
    """

    gini_gbm: float
    gini_glm: float
    gini_ratio: float
    deviance_ratio: float
    max_segment_deviation: float
    mean_segment_deviation: float
    n_segments: int

    def summary(self) -> str:
        """Return a compact human-readable summary string."""
        lines = [
            f"Gini (GBM):              {self.gini_gbm:.4f}",
            f"Gini (GLM surrogate):    {self.gini_glm:.4f}",
            f"Gini ratio:              {self.gini_ratio:.1%}",
            f"Deviance ratio:          {self.deviance_ratio:.4f}",
            f"Max segment deviation:   {self.max_segment_deviation:.1%}",
            f"Mean segment deviation:  {self.mean_segment_deviation:.1%}",
            f"Segments evaluated:      {self.n_segments:,}",
        ]
        return "\n".join(lines)


@dataclass
class DistillationReport:
    """
    Complete output of a :meth:`~insurance_distill.SurrogateGLM.report` call.

    Attributes
    ----------
    metrics : ValidationMetrics
        Scalar validation metrics.
    factor_tables : dict[str, pl.DataFrame]
        Mapping of feature name to its factor table DataFrame.
    lift_chart : pl.DataFrame or None
        Double-lift chart data (decile comparison of GBM vs GLM), or
        ``None`` if the surrogate has not been fitted yet.
    bin_specs : dict[str, BinSpec]
        The binning specification applied to each continuous feature.
    glm_coefficients : pl.DataFrame
        All fitted GLM coefficients with their feature names.
    """

    metrics: ValidationMetrics
    factor_tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    lift_chart: pl.DataFrame | None = None
    bin_specs: dict[str, BinSpec] = field(default_factory=dict)
    glm_coefficients: pl.DataFrame | None = None

    def __repr__(self) -> str:
        return (
            f"DistillationReport(\n"
            f"  gini_ratio={self.metrics.gini_ratio:.1%},\n"
            f"  max_segment_deviation={self.metrics.max_segment_deviation:.1%},\n"
            f"  features={list(self.factor_tables.keys())}\n"
            f")"
        )
