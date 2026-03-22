"""
SurrogateGLM: fit a multiplicative GLM on GBM pseudo-predictions.

The core insight is simple: generate predictions from your GBM, treat them
as the target for a GLM, and fit the GLM using those pseudo-predictions as
the response (with the actual exposure as weights). The resulting GLM factor
tables can then be loaded directly into a rating engine.

Design decisions
----------------
- We use glum (not statsmodels) because it is significantly faster, supports
  L1/L2 regularisation natively, and produces the same coefficient estimates
  as statsmodels for the unregularised case. It also handles large insurance
  datasets without running out of memory.
- The GLM is fitted in log-link space (Poisson or Gamma family). This means
  the factor tables are multiplicative by construction.
- We encode all features as one-hot dummy variables (reference-coded) before
  passing to glum. This avoids any assumptions about within-bin ordering.
- Interaction terms are created as simple cross-products of the binned
  one-hot columns, following the standard rating engine convention.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import polars as pl

from ._binning import BinMethod, OptimalBinner
from ._types import BinSpec, DistillationReport, ValidationMetrics
from ._validation import compute_gini, compute_deviance_ratio, compute_segment_deviation
from ._export import build_factor_tables, build_glm_coefficients_df

Family = Literal["poisson", "gamma", "tweedie"]


class SurrogateGLM:
    """
    Distil a fitted GBM into a multiplicative GLM suitable for a rating engine.

    Parameters
    ----------
    model :
        Any fitted model with a scikit-learn-compatible ``predict`` method.
        For CatBoost, pass the fitted ``CatBoostRegressor`` or
        ``CatBoostClassifier`` directly. For classification models, the
        GLM is fitted on predicted probabilities rather than class labels.
    X_train :
        Training data as a Polars DataFrame. All features used in the GBM
        must be present here.
    y_train :
        Actual target values (claims, loss amounts, etc.). Used only for
        computing validation metrics against the actual observations, not
        for fitting the surrogate.
    exposure :
        1-D array of exposure values (e.g. earned car-years). If None, all
        observations are assumed to have unit exposure.
    family : {"poisson", "gamma", "tweedie"}
        GLM family. Use ``"poisson"`` for frequency models and ``"gamma"``
        for severity models. The link function is always log.
    predict_method : str or None
        Name of the prediction method on the model. Defaults to
        ``"predict"`` for most sklearn models. Set to
        ``"predict_proba"`` for classifiers, or ``"predict"`` for
        CatBoost regressors. If None, we try ``"predict"`` first, then
        ``"predict_proba"``.
    alpha : float
        L2 regularisation strength passed to glum. Default is 0.0
        (unregularised). Increase if the GLM is overfitting to the binned
        data.

    Examples
    --------
    >>> surrogate = SurrogateGLM(
    ...     model=fitted_catboost,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     exposure=exposure_arr,
    ...     family="poisson",
    ... )
    >>> surrogate.fit(max_bins=10)
    >>> report = surrogate.report()
    >>> print(report.metrics.summary())
    """

    def __init__(
        self,
        model: Any,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        exposure: np.ndarray | None = None,
        family: Family = "poisson",
        predict_method: str | None = None,
        alpha: float = 0.0,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}.")

        self.model = model
        self.X_train = X_train
        self.y_train = np.asarray(y_train, dtype=float)
        self.exposure = (
            np.ones(len(X_train), dtype=float)
            if exposure is None
            else np.asarray(exposure, dtype=float)
        )
        self.family = family
        self.alpha = alpha

        self._predict_method = predict_method
        self._fitted = False
        self._bin_specs: dict[str, BinSpec] = {}
        self._features: list[str] = []
        self._categorical_features: list[str] = []
        self._glm = None
        self._design_col_names: list[str] = []
        self._pseudo_predictions: np.ndarray | None = None
        self._glm_predictions: np.ndarray | None = None
        self._intercept: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        max_bins: int = 10,
        binning_method: BinMethod = "tree",
        method_overrides: dict[str, BinMethod] | None = None,
        min_bin_size: int | float = 0.01,
        interaction_pairs: list[tuple[str, str]] | None = None,
    ) -> "SurrogateGLM":
        """
        Bin continuous features, then fit the surrogate GLM.

        Parameters
        ----------
        features :
            Continuous feature names to bin and include in the GLM. If None,
            all numeric columns in ``X_train`` are used.
        categorical_features :
            Feature names that should be used as-is (no binning). These must
            be string or integer columns.
        max_bins :
            Maximum bins per continuous feature.
        binning_method :
            Default binning method for continuous features.
        method_overrides :
            Per-feature binning method overrides.
        min_bin_size :
            Minimum observations per bin.
        interaction_pairs :
            Pairs of (already-binned) feature names for which to add
            interaction terms in the GLM.

        Returns
        -------
        SurrogateGLM
            Returns self for method chaining.
        """
        # Step 1: generate pseudo-predictions from the GBM
        self._pseudo_predictions = self._get_predictions()

        # Step 2: decide which features to bin
        if features is None:
            features = [
                c for c in self.X_train.columns
                if self.X_train[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
            ]
        self._features = features
        self._categorical_features = categorical_features or []

        # Step 3: bin continuous features
        binner = OptimalBinner(
            max_bins=max_bins,
            method=binning_method,
            min_bin_size=min_bin_size,
        )
        self._bin_specs = binner.fit(
            self.X_train,
            self._pseudo_predictions,
            features=self._features,
            weights=self.exposure,
            method_overrides=method_overrides,
        )

        # Step 4: build design matrix
        X_binned = binner.transform(self.X_train, self._bin_specs)
        X_design, col_names = self._build_design_matrix(
            X_binned, interaction_pairs=interaction_pairs or []
        )
        self._design_col_names = col_names

        # Step 5: fit the GLM with glum
        self._glm, self._intercept = self._fit_glm(X_design, self._pseudo_predictions)
        self._glm_predictions = self._predict_glm(X_design)
        self._fitted = True

        return self

    def report(self) -> DistillationReport:
        """
        Compute and return the full :class:`~insurance_distill._types.DistillationReport`.

        Must be called after :meth:`fit`.
        """
        self._check_fitted()

        gini_gbm = compute_gini(self._pseudo_predictions, self.exposure)
        gini_glm = compute_gini(self._glm_predictions, self.exposure)
        gini_ratio = gini_glm / gini_gbm if gini_gbm > 0 else float("nan")

        dev_ratio = compute_deviance_ratio(
            self._pseudo_predictions, self._glm_predictions, self.family
        )

        max_dev, mean_dev, n_seg = compute_segment_deviation(
            X_binned=self._get_binned_df(),
            pseudo=self._pseudo_predictions,
            glm_pred=self._glm_predictions,
            exposure=self.exposure,
            bin_features=[f"{f}__bin" for f in self._features],
            cat_features=self._categorical_features,
        )

        metrics = ValidationMetrics(
            gini_gbm=gini_gbm,
            gini_glm=gini_glm,
            gini_ratio=gini_ratio,
            deviance_ratio=dev_ratio,
            max_segment_deviation=max_dev,
            mean_segment_deviation=mean_dev,
            n_segments=n_seg,
        )

        factor_tables = build_factor_tables(
            glm=self._glm,
            bin_specs=self._bin_specs,
            col_names=self._design_col_names,
            intercept=self._intercept,
            cat_features=self._categorical_features,
        )

        coeff_df = build_glm_coefficients_df(
            glm=self._glm,
            col_names=self._design_col_names,
            intercept=self._intercept,
        )

        lift_chart = self._build_lift_chart()

        return DistillationReport(
            metrics=metrics,
            factor_tables=factor_tables,
            lift_chart=lift_chart,
            bin_specs=self._bin_specs,
            glm_coefficients=coeff_df,
        )

    def factor_table(self, feature: str) -> pl.DataFrame:
        """
        Return the factor table for a single feature.

        Returns a DataFrame with columns:
        - ``level``: the bin label or category value
        - ``relativity``: the multiplicative factor relative to the base level
        - ``base_factor``: the raw GLM coefficient on the log scale
        - ``n_obs``: number of training observations in this level

        Parameters
        ----------
        feature :
            Feature name (not the ``__bin`` suffixed name).
        """
        self._check_fitted()
        report = self.report()
        if feature not in report.factor_tables:
            available = list(report.factor_tables.keys())
            raise KeyError(
                f"Feature '{feature}' not found in factor tables. "
                f"Available: {available}"
            )
        return report.factor_tables[feature]

    def export_csv(
        self,
        directory: str,
        prefix: str = "",
        include_base: bool = True,
    ) -> list[str]:
        """
        Export one CSV file per feature factor table.

        Parameters
        ----------
        directory :
            Output directory. Will be created if it does not exist.
        prefix :
            Optional filename prefix, e.g. ``"motor_freq_"``.
        include_base :
            If True, include a ``base.csv`` with the model intercept /
            base factor.

        Returns
        -------
        list[str]
            List of file paths written.
        """
        import os

        self._check_fitted()
        os.makedirs(directory, exist_ok=True)
        report = self.report()

        written = []
        for feat, df in report.factor_tables.items():
            path = os.path.join(directory, f"{prefix}{feat}.csv")
            df.write_csv(path)
            written.append(path)

        if include_base:
            base_df = pl.DataFrame(
                {
                    "intercept_log": [self._intercept],
                    "base_factor": [float(np.exp(self._intercept))],
                }
            )
            path = os.path.join(directory, f"{prefix}base.csv")
            base_df.write_csv(path)
            written.append(path)

        return written

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "SurrogateGLM has not been fitted yet. Call .fit() first."
            )

    def _get_predictions(self) -> np.ndarray:
        """Dispatch to the correct prediction method on the model."""
        method_name = self._predict_method

        if method_name is None:
            # Auto-detect
            if hasattr(self.model, "predict"):
                method_name = "predict"
            elif hasattr(self.model, "predict_proba"):
                method_name = "predict_proba"
            else:
                raise AttributeError(
                    "Model has neither 'predict' nor 'predict_proba' method."
                )

        method = getattr(self.model, method_name)

        # Try Polars first; fall back to pandas if the model requires it
        try:
            preds = method(self.X_train)
        except Exception:
            try:
                preds = method(self.X_train.to_pandas())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate predictions from model using method '{method_name}'. "
                    f"Original error: {e}"
                ) from e

        preds = np.asarray(preds, dtype=float)
        if preds.ndim == 2:
            # predict_proba returns (n, 2) for binary classifiers
            preds = preds[:, 1]

        # Clip to a small positive value for log-link models
        preds = np.clip(preds, 1e-8, None)
        return preds

    def _build_design_matrix(
        self,
        X_binned: pl.DataFrame,
        interaction_pairs: list[tuple[str, str]],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build a one-hot encoded design matrix from the binned DataFrame.

        Reference coding: the first level of each feature is dropped to avoid
        perfect multicollinearity. This matches the default in Emblem/Radar.

        Returns
        -------
        X_design : np.ndarray of shape (n, p)
        col_names : list[str] with p names, one per column
        """
        bin_cols = [f"{f}__bin" for f in self._features if f"{f}__bin" in X_binned.columns]
        cat_cols = [c for c in self._categorical_features if c in X_binned.columns]

        all_factor_cols = bin_cols + cat_cols
        col_blocks: list[np.ndarray] = []
        col_names: list[str] = []

        for col in all_factor_cols:
            series = X_binned[col].cast(pl.String)
            levels = sorted(series.unique().to_list())
            if len(levels) <= 1:
                continue
            # Drop first level (reference category)
            for level in levels[1:]:
                indicator = (series == level).cast(pl.Int8).to_numpy()
                col_blocks.append(indicator)
                col_names.append(f"{col}={level}")

        # Interaction terms
        for feat_a, feat_b in interaction_pairs:
            col_a = f"{feat_a}__bin" if feat_a in self._features else feat_a
            col_b = f"{feat_b}__bin" if feat_b in self._features else feat_b
            if col_a not in X_binned.columns or col_b not in X_binned.columns:
                warnings.warn(
                    f"Interaction pair ({feat_a}, {feat_b}) skipped - "
                    "one or both features not found in binned DataFrame.",
                    stacklevel=3,
                )
                continue
            s_a = X_binned[col_a].cast(pl.String)
            s_b = X_binned[col_b].cast(pl.String)
            lvls_a = sorted(s_a.unique().to_list())[1:]  # skip reference
            lvls_b = sorted(s_b.unique().to_list())[1:]  # skip reference
            for la in lvls_a:
                for lb in lvls_b:
                    indicator = (
                        ((s_a == la) & (s_b == lb)).cast(pl.Int8).to_numpy()
                    )
                    col_blocks.append(indicator)
                    col_names.append(f"{col_a}={la}:{col_b}={lb}")

        if not col_blocks:
            raise ValueError(
                "Design matrix is empty - no valid features to include in the GLM. "
                "Check that the features list is correct and the data contains "
                "more than one distinct value per feature."
            )

        X_design = np.column_stack(col_blocks).astype(np.float64)
        return X_design, col_names

    def _fit_glm(
        self,
        X_design: np.ndarray,
        pseudo: np.ndarray,
    ) -> tuple[Any, float]:
        """
        Fit the GLM using glum and return (fitted_glm, log_intercept).
        """
        from glum import GeneralizedLinearRegressor

        family_map = {
            "poisson": "poisson",
            "gamma": "gamma",
            "tweedie": "tweedie",
        }
        glum_family = family_map.get(self.family)
        if glum_family is None:
            raise ValueError(f"Unknown family: {self.family!r}")

        glm = GeneralizedLinearRegressor(
            family=glum_family,
            link="log",
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=200,
        )
        glm.fit(X_design, pseudo, sample_weight=self.exposure)

        return glm, float(glm.intercept_)

    def _predict_glm(self, X_design: np.ndarray) -> np.ndarray:
        return np.asarray(self._glm.predict(X_design), dtype=float)

    def _get_binned_df(self) -> pl.DataFrame:
        """Re-create the binned DataFrame (needed for validation)."""
        from ._binning import OptimalBinner

        binner = OptimalBinner()
        return binner.transform(self.X_train, self._bin_specs)

    def _build_lift_chart(self) -> "pl.DataFrame":
        """Build a double-lift chart comparing GBM pseudo-predictions to GLM."""
        from ._validation import double_lift_chart

        return double_lift_chart(
            self._pseudo_predictions,
            self._glm_predictions,
            exposure=self.exposure,
            n_deciles=10,
        )
