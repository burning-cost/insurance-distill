"""
LassoGuidedGLM: partial-dependence-guided binning with lasso feature selection.

This implements the approach described in Lindholm & Palmquist (2024),
"Black-box Guided GLM Building" (SSRN 4691626). The core idea is to let the
GBM tell you where the bin boundaries should fall — via its partial dependence
functions — rather than imposing arbitrary quantile bins. Lasso then selects
which bins actually matter, and a final unpenalised refit ensures the GLM is
properly calibrated.

Design decisions
----------------
- Partial dependence is computed using sklearn's ``partial_dependence`` with
  method="brute", which works for any black-box model (not just tree ensembles).
  The "recursion" method is faster for tree ensembles but is not universally
  available.
- Split points are detected as local extrema and inflection points on the PD
  curve, not by recursive variance partitioning. This is closer to the paper's
  intent: the GBM's predicted response shape directly determines where bin
  boundaries fall.
- The lasso is fitted via glum's GeneralizedLinearRegressor with the L1 penalty
  (``l1_ratio=1.0``). This is the elastic-net parametrisation, so alpha controls
  overall penalty strength.
- The calibration refit uses alpha=0.0 on the selected columns only, so the
  GLM is fully unpenalised and the predicted totals match the actual target.
- We deliberately do not re-run PD extraction at predict time — the bin edges
  are fixed at fit time and applied deterministically to new data. This is the
  only sensible choice for a rating engine that needs reproducible factors.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import polars as pl

Family = Literal["poisson", "gamma", "tweedie"]


def _pd_guided_splits(
    model: Any,
    X_numpy: np.ndarray,
    feature_idx: int,
    feature_name: str,
    n_splits: int,
    grid_resolution: int = 100,
) -> list[float]:
    """
    Compute partial dependence for one feature and return split point candidates.

    Split candidates are chosen at local extrema and inflection points of the PD
    curve, capped at ``n_splits``. If the PD curve is monotone (no local extrema),
    we fall back to evenly-spaced quantile splits over the PD grid.

    Parameters
    ----------
    model :
        Fitted model with a scikit-learn-compatible interface.
    X_numpy :
        Training data as a float numpy array, shape (n, p).
    feature_idx :
        Column index of the feature in X_numpy.
    feature_name :
        Feature name (for warning messages only).
    n_splits :
        Maximum number of interior split points to return.
    grid_resolution :
        Number of evenly-spaced grid points to evaluate PD at. Higher values
        give smoother curves but increase computation time.

    Returns
    -------
    list[float]
        Sorted list of interior cut-point values (not including -inf / +inf).
    """
    from sklearn.inspection import partial_dependence

    try:
        pd_result = partial_dependence(
            model,
            X=X_numpy,
            features=[feature_idx],
            kind="average",
            grid_resolution=grid_resolution,
            method="brute",
        )
    except Exception as exc:
        warnings.warn(
            f"partial_dependence failed for feature '{feature_name}': {exc}. "
            "Falling back to quantile splits.",
            stacklevel=4,
        )
        col = X_numpy[:, feature_idx]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            return []
        quantiles = np.linspace(0, 100, n_splits + 2)[1:-1]
        return sorted(np.unique(np.percentile(col, quantiles)).tolist())

    pd_values = pd_result["average"][0]   # shape (grid_resolution,)
    grid_values = pd_result["grid_values"][0]  # shape (grid_resolution,)

    if len(grid_values) < 3:
        return []

    # Detect local extrema: points where the derivative changes sign
    diff = np.diff(pd_values)
    sign_changes = np.where(np.diff(np.sign(diff)))[0] + 1  # indices in grid_values

    # Detect inflection points: second derivative changes sign
    second_diff = np.diff(diff)
    inflection_changes = np.where(np.diff(np.sign(second_diff)))[0] + 2

    candidate_indices = np.union1d(sign_changes, inflection_changes)

    if len(candidate_indices) == 0:
        # Monotone curve — fall back to evenly-spaced quantile cuts on the
        # actual data distribution, but honour the n_splits limit
        col = X_numpy[:, feature_idx]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            return []
        quantiles = np.linspace(0, 100, n_splits + 2)[1:-1]
        return sorted(np.unique(np.percentile(col, quantiles)).tolist())

    # Select up to n_splits candidates, spread across the curve
    if len(candidate_indices) > n_splits:
        selection = np.linspace(0, len(candidate_indices) - 1, n_splits, dtype=int)
        candidate_indices = candidate_indices[selection]

    # Convert grid indices to actual feature values
    cuts = [float(grid_values[i]) for i in candidate_indices]

    # Ensure cuts are within the observed data range (grid may extend beyond it)
    col = X_numpy[:, feature_idx]
    col = col[np.isfinite(col)]
    lo, hi = float(col.min()), float(col.max())
    cuts = [c for c in cuts if lo < c < hi]

    return sorted(set(cuts))


def _apply_cuts(series: pl.Series, cuts: list[float], feature: str) -> pl.Series:
    """
    Bin a Polars Series of floats into labelled string intervals.

    Parameters
    ----------
    series :
        Float series to bin.
    cuts :
        Sorted interior cut-points (no -inf / +inf).
    feature :
        Feature name, used only for label construction.

    Returns
    -------
    pl.Series
        String series of bin labels, named ``<feature>__bin``.
    """
    edges = [-np.inf] + list(cuts) + [np.inf]

    def _label(lo: float, hi: float) -> str:
        lo_s = "-inf" if np.isinf(lo) and lo < 0 else f"{lo:.2f}"
        hi_s = "+inf" if np.isinf(hi) and hi > 0 else f"{hi:.2f}"
        return f"[{lo_s}, {hi_s})"

    labels = [_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    interior = [e for e in edges if not np.isinf(e)]
    result = series.cast(pl.Float64).cut(interior, labels=labels, left_closed=True)
    return result.cast(pl.String).alias(f"{feature}__bin")


def _build_ohe_block(
    series: pl.Series,
    col_name: str,
) -> tuple[np.ndarray, list[str], str]:
    """
    One-hot encode a string series with reference (first level) dropped.

    Parameters
    ----------
    series :
        String series to encode.
    col_name :
        Column name used to construct dummy column names.

    Returns
    -------
    block : np.ndarray, shape (n, k-1)
    col_names : list of str, length k-1
    base_level : str — the reference level that was dropped
    """
    levels = sorted(series.unique().to_list())
    if len(levels) <= 1:
        return np.empty((len(series), 0), dtype=np.float64), [], levels[0] if levels else ""

    base_level = levels[0]
    cols = []
    names = []
    for lv in levels[1:]:
        cols.append((series == lv).cast(pl.Int8).to_numpy().astype(np.float64))
        names.append(f"{col_name}={lv}")

    block = np.column_stack(cols)
    return block, names, base_level


class LassoGuidedGLM:
    """
    Distil a fitted GBM into a multiplicative GLM using partial-dependence-guided
    binning and lasso feature selection.

    Unlike :class:`SurrogateGLM`, which bins features using a decision tree on
    GBM predictions, this class uses the GBM's partial dependence curves to place
    bin boundaries at points where the GBM's learned response actually changes
    direction. Lasso then automatically selects which bins are large enough to
    warrant inclusion in the final GLM.

    The fitting pipeline has three stages:

    1. For each feature, compute the GBM's partial dependence curve and detect
       split candidates at local extrema and inflection points.
    2. Encode all features as dummy variables and fit a lasso GLM. This performs
       simultaneous feature selection and coefficient estimation.
    3. Identify which features survived lasso selection, then refit without any
       penalty on those features only. This calibration step ensures that the
       GLM's predicted totals match the actual target within each cell.

    Parameters
    ----------
    gbm_model :
        Fitted model with a scikit-learn-compatible ``predict`` method.
        Must accept a 2-D numpy array as input.
    feature_names :
        Names of the continuous features to include. Order must match the
        columns that the model expects.
    n_bins : int
        Maximum number of PD-guided split points per feature (i.e. the maximum
        number of bins is ``n_bins + 1``).
    alpha : float
        Lasso penalty strength passed to glum. Higher values produce sparser
        solutions. The default of 1.0 is a reasonable starting point; tune
        downward if too many features are dropped.
    family : {"poisson", "gamma", "tweedie"}
        GLM response distribution. Use ``"poisson"`` for frequency models and
        ``"gamma"`` for severity.
    power : float
        Tweedie power parameter. Ignored unless ``family="tweedie"``. Common
        values: 1.0 (Poisson), 1.5 (compound Poisson-Gamma), 2.0 (Gamma).
    pd_grid_resolution : int
        Number of grid points used to evaluate each partial dependence curve.
        Increase for smoother curves on complex models; decrease for speed.
    """

    def __init__(
        self,
        gbm_model: Any,
        feature_names: list[str],
        n_bins: int = 10,
        alpha: float = 1.0,
        family: Family = "tweedie",
        power: float = 1.5,
        pd_grid_resolution: int = 100,
    ) -> None:
        if n_bins <= 0:
            raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}.")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}.")
        if pd_grid_resolution <= 0:
            raise ValueError(f"pd_grid_resolution must be a positive integer, got {pd_grid_resolution!r}.")

        self.gbm_model = gbm_model
        self.feature_names = list(feature_names)
        self.n_bins = n_bins
        self.alpha = alpha
        self.family = family
        self.power = power
        self.pd_grid_resolution = pd_grid_resolution

        # Set after fit()
        self._fitted: bool = False
        self._cuts: dict[str, list[float]] = {}
        self._selected_features: list[str] = []
        self._lasso_col_names: list[str] = []
        self._final_col_names: list[str] = []
        self._base_levels: dict[str, str] = {}
        self._lasso_glm: Any = None
        self._final_glm: Any = None
        self._intercept: float = 0.0
        self._X_binned: pl.DataFrame | None = None
        self._feature_col_map: dict[str, str] = {}  # feature -> __bin col name
        self._lasso_coef_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> "LassoGuidedGLM":
        """
        Run the full PD-guided lasso pipeline.

        Parameters
        ----------
        X :
            Training data as a Polars DataFrame. Must contain all columns listed
            in ``feature_names``.
        y :
            Observed target values (claims counts, loss amounts, etc.).
        sample_weight :
            Observation weights. If both ``sample_weight`` and ``exposure`` are
            provided, they are multiplied together.
        exposure :
            Exposure values. When provided, the GLM is fitted with an offset of
            ``log(exposure)`` so that predicted values are rates.

        Returns
        -------
        LassoGuidedGLM
            Returns self for method chaining.
        """
        y = np.asarray(y, dtype=float)
        n = len(X)

        # Resolve weights: combine sample_weight and exposure
        if sample_weight is not None and exposure is not None:
            weights = np.asarray(sample_weight, dtype=float) * np.asarray(exposure, dtype=float)
        elif sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
        elif exposure is not None:
            weights = np.asarray(exposure, dtype=float)
        else:
            weights = np.ones(n, dtype=float)

        # Stage 1: PD-guided binning
        X_numpy = X.select(self.feature_names).to_numpy().astype(float)
        self._cuts = {}
        for i, feat in enumerate(self.feature_names):
            cuts = _pd_guided_splits(
                model=self.gbm_model,
                X_numpy=X_numpy,
                feature_idx=i,
                feature_name=feat,
                n_splits=self.n_bins,
                grid_resolution=self.pd_grid_resolution,
            )
            self._cuts[feat] = cuts

        # Apply bins to produce binned DataFrame
        binned_cols = {feat: _apply_cuts(X[feat], self._cuts[feat], feat)
                       for feat in self.feature_names}
        self._X_binned = X.with_columns(list(binned_cols.values()))
        self._feature_col_map = {feat: f"{feat}__bin" for feat in self.feature_names}

        # Stage 2: build lasso design matrix and fit
        X_lasso, lasso_col_names, base_levels = self._build_design_matrix(self.feature_names)
        self._lasso_col_names = lasso_col_names
        self._base_levels = base_levels

        self._lasso_glm = self._fit_glm(
            X_lasso, y, weights=weights, alpha=self.alpha, l1_ratio=1.0
        )
        self._lasso_coef_ = np.asarray(self._lasso_glm.coef_, dtype=float)

        # Identify which features survived lasso (at least one non-zero coefficient)
        selected = self._selected_features_from_lasso(lasso_col_names, self._lasso_coef_)
        self._selected_features = selected

        if not selected:
            warnings.warn(
                "Lasso selected no features. Try reducing alpha. "
                "Returning an intercept-only model.",
                stacklevel=2,
            )
            self._final_glm = self._lasso_glm
            self._final_col_names = []
            self._intercept = float(self._lasso_glm.intercept_)
            self._fitted = True
            return self

        # Stage 3: calibration refit on selected features only
        X_final, final_col_names, _ = self._build_design_matrix(selected)
        self._final_col_names = final_col_names

        self._final_glm = self._fit_glm(
            X_final, y, weights=weights, alpha=0.0, l1_ratio=0.0
        )
        self._intercept = float(self._final_glm.intercept_)
        self._fitted = True
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Generate predictions from the distilled GLM.

        Parameters
        ----------
        X :
            Input data as a Polars DataFrame. Must contain all columns in
            ``feature_names``.

        Returns
        -------
        np.ndarray
            1-D array of predicted values on the response scale.
        """
        self._check_fitted()

        if not self._selected_features:
            # Intercept-only: predict the mean
            n = len(X)
            return np.full(n, float(np.exp(self._intercept)))

        # Bin the input data using the fitted cut-points
        binned_cols = [_apply_cuts(X[feat], self._cuts[feat], feat)
                       for feat in self._selected_features
                       if feat in X.columns]
        X_binned = X.with_columns(binned_cols)

        X_design, _, _ = self._build_design_matrix(
            self._selected_features, X_binned=X_binned
        )
        return np.asarray(self._final_glm.predict(X_design), dtype=float)

    def factor_tables(self) -> dict[str, pl.DataFrame]:
        """
        Return factor tables for all selected features.

        Each table has columns:

        - ``level``: bin label (string interval)
        - ``log_coefficient``: raw GLM coefficient on the log scale; 0.0 for the
          reference (first) level
        - ``relativity``: multiplicative factor, i.e. ``exp(log_coefficient)``

        Returns
        -------
        dict[str, pl.DataFrame]
            One entry per selected feature.
        """
        self._check_fitted()

        coef = np.asarray(self._final_glm.coef_, dtype=float)
        coef_map: dict[str, float] = dict(zip(self._final_col_names, coef))

        tables: dict[str, pl.DataFrame] = {}
        for feat in self._selected_features:
            col_name = f"{feat}__bin"
            base_level = self._base_levels.get(feat, "")
            cuts = self._cuts.get(feat, [])
            edges = [-np.inf] + list(cuts) + [np.inf]

            def _label(lo: float, hi: float) -> str:
                lo_s = "-inf" if np.isinf(lo) and lo < 0 else f"{lo:.2f}"
                hi_s = "+inf" if np.isinf(hi) and hi > 0 else f"{hi:.2f}"
                return f"[{lo_s}, {hi_s})"

            all_labels = [_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            rows = []
            for label in all_labels:
                col_key = f"{col_name}={label}"
                log_coef = coef_map.get(col_key, 0.0)
                rows.append(
                    {
                        "level": label,
                        "log_coefficient": log_coef,
                        "relativity": float(np.exp(log_coef)),
                    }
                )
            tables[feat] = pl.DataFrame(rows)

        return tables

    def summary(self) -> None:
        """
        Print feature selection results and calibration diagnostics to stdout.
        """
        self._check_fitted()

        all_feats = set(self.feature_names)
        selected = set(self._selected_features)
        dropped = all_feats - selected

        lasso_coef = self._lasso_coef_ if self._lasso_coef_ is not None else np.array([])
        n_nonzero = int(np.sum(lasso_coef != 0.0))

        print("=" * 60)
        print("LassoGuidedGLM — Fit Summary")
        print("=" * 60)
        print(f"Family:             {self.family}" + (f" (power={self.power})" if self.family == "tweedie" else ""))
        print(f"Lasso alpha:        {self.alpha}")
        print(f"Features input:     {len(all_feats)}")
        print(f"Features selected:  {len(selected)}")
        print(f"Features dropped:   {len(dropped)}")
        print(f"Non-zero lasso cols:{n_nonzero} / {len(self._lasso_col_names)}")
        print(f"Intercept (log):    {self._intercept:.4f}  (exp = {np.exp(self._intercept):.4f})")
        print()

        if selected:
            print("Selected features:")
            for feat in sorted(selected):
                n_cuts = len(self._cuts.get(feat, []))
                print(f"  {feat:30s}  {n_cuts + 1} bins")

        if dropped:
            print()
            print("Dropped by lasso:")
            for feat in sorted(dropped):
                print(f"  {feat}")

        print("=" * 60)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "LassoGuidedGLM has not been fitted yet. Call .fit() first."
            )

    def _glum_family(self) -> Any:
        """Return the glum family object for the configured family."""
        from glum import TweedieDistribution

        if self.family == "poisson":
            return "poisson"
        elif self.family == "gamma":
            return "gamma"
        elif self.family == "tweedie":
            return TweedieDistribution(power=self.power)
        else:
            raise ValueError(f"Unknown family: {self.family!r}. Choose 'poisson', 'gamma', or 'tweedie'.")

    def _build_design_matrix(
        self,
        features: list[str],
        X_binned: pl.DataFrame | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, str]]:
        """
        Build a one-hot encoded design matrix for the given features.

        Parameters
        ----------
        features :
            Feature names to include. Must be a subset of ``self.feature_names``.
        X_binned :
            Pre-binned DataFrame. If None, uses ``self._X_binned``.

        Returns
        -------
        X_design : np.ndarray, shape (n, p)
        col_names : list[str], length p
        base_levels : dict[str, str] — reference level per feature
        """
        df = X_binned if X_binned is not None else self._X_binned
        if df is None:
            raise RuntimeError("No binned DataFrame available. Call fit() first.")

        blocks: list[np.ndarray] = []
        names: list[str] = []
        base_levels: dict[str, str] = {}

        for feat in features:
            col_name = f"{feat}__bin"
            if col_name not in df.columns:
                warnings.warn(
                    f"Binned column '{col_name}' not found in DataFrame. Skipping.",
                    stacklevel=3,
                )
                continue
            series = df[col_name].cast(pl.String)
            block, col_names_feat, base_lv = _build_ohe_block(series, col_name)
            if block.shape[1] == 0:
                continue
            blocks.append(block)
            names.extend(col_names_feat)
            base_levels[feat] = base_lv

        if not blocks:
            raise ValueError(
                "Design matrix is empty. Check that the selected features have "
                "more than one distinct bin value."
            )

        X_design = np.column_stack(blocks).astype(np.float64)
        return X_design, names, base_levels

    def _fit_glm(
        self,
        X_design: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        alpha: float,
        l1_ratio: float,
    ) -> Any:
        """
        Fit a GLM using glum.

        Parameters
        ----------
        X_design :
            Design matrix.
        y :
            Response variable.
        weights :
            Observation weights.
        alpha :
            Regularisation strength. 0.0 = unpenalised.
        l1_ratio :
            Elastic net mixing parameter. 1.0 = pure lasso, 0.0 = ridge.

        Returns
        -------
        Fitted GeneralizedLinearRegressor instance.
        """
        from glum import GeneralizedLinearRegressor

        glm = GeneralizedLinearRegressor(
            family=self._glum_family(),
            link="log",
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=500,
        )
        glm.fit(X_design, y, sample_weight=weights)
        return glm

    def _selected_features_from_lasso(
        self,
        col_names: list[str],
        coef: np.ndarray,
    ) -> list[str]:
        """
        Identify features that have at least one non-zero lasso coefficient.

        Parameters
        ----------
        col_names :
            Design matrix column names (``<feature>__bin=<level>`` format).
        coef :
            Fitted lasso coefficients.

        Returns
        -------
        list[str]
            Feature names (without ``__bin`` suffix) with at least one selected bin.
        """
        active_features: set[str] = set()
        for name, c in zip(col_names, coef):
            if c != 0.0:
                # col_name is "<feature>__bin=<level>", extract feature
                if "__bin=" in name:
                    feat = name.split("__bin=")[0]
                    active_features.add(feat)

        # Preserve original order
        return [f for f in self.feature_names if f in active_features]
