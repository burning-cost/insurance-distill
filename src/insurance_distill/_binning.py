"""
Optimal binning of continuous variables for GLM factor construction.

The goal is to produce a small number of homogeneous bins such that the GLM
factor table for each variable is both interpretable and captures the shape of
the GBM's learned response.

Three strategies are available:

tree
    Fits a single-variable decision tree (CART) on the GBM pseudo-predictions.
    This is the default and produces bins that are optimal in the sense of
    minimising within-bin variance of the target. It naturally respects
    monotonicity where the data supports it, and produces splits at
    statistically meaningful thresholds.

quantile
    Divides the variable into quantile bands. Simple and robust, but ignores
    the GBM's actual response shape. Useful as a fallback when the tree
    produces degenerate splits.

isotonic
    Fits an isotonic regression on the GBM response and uses its change-points
    as split locations. Good for variables where the GBM response is
    monotone (e.g. no-claims discount).
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeRegressor

from ._types import BinSpec

BinMethod = Literal["tree", "quantile", "isotonic"]


def _format_bin_label(lo: float, hi: float, decimals: int = 2) -> str:
    """Build a human-readable interval label such as ``[10.00, 25.00)``."""
    lo_str = "-inf" if np.isinf(lo) and lo < 0 else f"{lo:.{decimals}f}"
    hi_str = "+inf" if np.isinf(hi) and hi > 0 else f"{hi:.{decimals}f}"
    return f"[{lo_str}, {hi_str})"


def _edges_to_spec(
    feature: str,
    interior_cuts: list[float],
    method: str,
    decimals: int = 2,
) -> BinSpec:
    """
    Convert a list of interior cut-points to a :class:`BinSpec`.

    Parameters
    ----------
    feature :
        Column name.
    interior_cuts :
        Sorted list of interior cut-points (no -inf / +inf sentinels).
    method :
        Binning method label.
    decimals :
        Decimal precision for label formatting.
    """
    cuts = sorted(set(interior_cuts))
    edges = [-np.inf] + cuts + [np.inf]
    labels = [
        _format_bin_label(edges[i], edges[i + 1], decimals)
        for i in range(len(edges) - 1)
    ]
    return BinSpec(
        feature=feature,
        bin_edges=edges,
        bin_labels=labels,
        method=method,
        n_bins=len(labels),
    )


class OptimalBinner:
    """
    Bin one or more continuous variables using GBM predictions as a guide.

    Parameters
    ----------
    max_bins : int
        Maximum number of bins per variable. The tree-based method will use
        this as ``max_leaf_nodes``; the quantile method will cut into exactly
        this many bands (fewer if there are not enough distinct values).
    method : {"tree", "quantile", "isotonic"}
        Default binning method. Can be overridden per variable via
        :meth:`fit_feature`.
    min_bin_size : int or float
        Minimum number of observations per bin. If an int, treated as an
        absolute count. If a float in (0, 1), treated as a fraction of the
        total number of rows. Tree-based binning uses this via
        ``min_samples_leaf``.
    label_decimals : int
        Decimal places used when formatting bin edge labels.

    Examples
    --------
    >>> binner = OptimalBinner(max_bins=8, method="tree")
    >>> specs = binner.fit(X, gbm_predictions, features=["driver_age", "vehicle_value"])
    >>> X_binned = binner.transform(X, specs)
    """

    def __init__(
        self,
        max_bins: int = 10,
        method: BinMethod = "tree",
        min_bin_size: int | float = 0.01,
        label_decimals: int = 2,
    ) -> None:
        self.max_bins = max_bins
        self.method = method
        self.min_bin_size = min_bin_size
        self.label_decimals = label_decimals

    def _resolve_min_samples(self, n: int) -> int:
        if isinstance(self.min_bin_size, float):
            return max(1, int(self.min_bin_size * n))
        return max(1, self.min_bin_size)

    def fit_feature(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature: str,
        method: BinMethod | None = None,
        weights: np.ndarray | None = None,
    ) -> BinSpec:
        """
        Fit binning for a single feature.

        Parameters
        ----------
        x :
            1-D array of feature values (float).
        y :
            1-D array of GBM pseudo-predictions (the target for binning).
        feature :
            Column name, used only for labelling.
        method :
            Override the instance-level ``method``.
        weights :
            Optional sample weights (e.g. exposure). Used by the tree
            binner (``sample_weight`` argument) and isotonic binner.

        Returns
        -------
        BinSpec
        """
        m = method or self.method
        n = len(x)
        min_samples = self._resolve_min_samples(n)

        # Drop NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            warnings.warn(
                f"Feature '{feature}' has fewer than 2 valid observations. "
                "Returning a single bin.",
                stacklevel=2,
            )
            return _edges_to_spec(feature, [], m, self.label_decimals)

        x_clean = x[mask]
        y_clean = y[mask]
        w_clean = weights[mask] if weights is not None else None

        n_distinct = len(np.unique(x_clean))
        effective_bins = min(self.max_bins, n_distinct)

        if effective_bins <= 1:
            return _edges_to_spec(feature, [], m, self.label_decimals)

        if m == "tree":
            return self._fit_tree(
                x_clean, y_clean, feature, effective_bins, min_samples, w_clean
            )
        elif m == "quantile":
            return self._fit_quantile(x_clean, feature, effective_bins)
        elif m == "isotonic":
            return self._fit_isotonic(x_clean, y_clean, feature, w_clean)
        else:
            raise ValueError(f"Unknown binning method: {m!r}. Choose 'tree', 'quantile', or 'isotonic'.")

    def _fit_tree(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature: str,
        max_bins: int,
        min_samples: int,
        weights: np.ndarray | None,
    ) -> BinSpec:
        tree = DecisionTreeRegressor(
            max_leaf_nodes=max_bins,
            min_samples_leaf=min_samples,
        )
        tree.fit(x.reshape(-1, 1), y, sample_weight=weights)

        # Extract split thresholds from the tree's internal structure.
        # sklearn uses -2.0 as the sentinel for leaf nodes (TREE_LEAF constant).
        # Filter it out to get only genuine split thresholds.
        thresholds = tree.tree_.threshold
        interior_cuts = sorted(
            float(c) for c in thresholds if c != -2.0
        )

        return _edges_to_spec(feature, interior_cuts, "tree", self.label_decimals)

    def _fit_quantile(
        self,
        x: np.ndarray,
        feature: str,
        max_bins: int,
    ) -> BinSpec:
        quantiles = np.linspace(0, 100, max_bins + 1)[1:-1]
        cuts = np.unique(np.percentile(x, quantiles))
        return _edges_to_spec(feature, cuts.tolist(), "quantile", self.label_decimals)

    def _fit_isotonic(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature: str,
        weights: np.ndarray | None,
    ) -> BinSpec:
        """
        Use isotonic regression change-points as bin boundaries.

        We sort by x, fit isotonic regression, and find where the fitted
        value changes - those are the natural bin boundaries.
        """
        from sklearn.isotonic import IsotonicRegression

        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        w_sorted = weights[order] if weights is not None else None

        iso = IsotonicRegression(out_of_bounds="clip")
        y_iso = iso.fit_transform(x_sorted, y_sorted, sample_weight=w_sorted)

        # Find change-points: locations where the isotonic value changes
        changes = np.where(np.diff(y_iso) != 0)[0]
        if len(changes) == 0:
            return _edges_to_spec(feature, [], "isotonic", self.label_decimals)

        # Use midpoints between x values at change-points as cuts
        cuts = []
        for idx in changes:
            if idx + 1 < len(x_sorted):
                midpoint = (x_sorted[idx] + x_sorted[idx + 1]) / 2.0
                cuts.append(float(midpoint))

        # Limit to max_bins - 1 cuts
        if len(cuts) > self.max_bins - 1:
            # Keep the most evenly spaced cuts
            indices = np.linspace(0, len(cuts) - 1, self.max_bins - 1, dtype=int)
            cuts = [cuts[i] for i in indices]

        return _edges_to_spec(feature, sorted(set(cuts)), "isotonic", self.label_decimals)

    def fit(
        self,
        X: pl.DataFrame,
        predictions: np.ndarray,
        features: list[str],
        weights: np.ndarray | None = None,
        method_overrides: dict[str, BinMethod] | None = None,
    ) -> dict[str, BinSpec]:
        """
        Fit binning for multiple features at once.

        Parameters
        ----------
        X :
            Training data as a Polars DataFrame.
        predictions :
            1-D array of GBM pseudo-predictions (the quantity being distilled).
        features :
            List of column names to bin.
        weights :
            Optional sample weights aligned with rows of ``X``.
        method_overrides :
            Per-feature method overrides, e.g.
            ``{"driver_age": "isotonic", "vehicle_value": "quantile"}``.

        Returns
        -------
        dict[str, BinSpec]
            Mapping of feature name to its fitted :class:`BinSpec`.
        """
        overrides = method_overrides or {}
        specs: dict[str, BinSpec] = {}

        for feat in features:
            if feat not in X.columns:
                raise ValueError(f"Feature '{feat}' not found in DataFrame columns.")
            x_arr = X[feat].to_numpy().astype(float)
            m = overrides.get(feat)
            specs[feat] = self.fit_feature(x_arr, predictions, feat, method=m, weights=weights)

        return specs

    def transform(
        self,
        X: pl.DataFrame,
        specs: dict[str, BinSpec],
    ) -> pl.DataFrame:
        """
        Apply fitted bin specs to a DataFrame, replacing continuous columns
        with their string bin labels.

        Columns not in ``specs`` are passed through unchanged.

        Parameters
        ----------
        X :
            DataFrame to transform.
        specs :
            Fitted bin specs from :meth:`fit`.

        Returns
        -------
        pl.DataFrame
            DataFrame with binned columns added as ``<feature>__bin`` columns.
            Original numeric columns are preserved.
        """
        out = X.clone()
        for feat, spec in specs.items():
            if feat not in out.columns:
                continue
            binned = spec.apply(out[feat].cast(pl.Float64))
            out = out.with_columns(binned.cast(pl.String).alias(f"{feat}__bin"))
        return out
