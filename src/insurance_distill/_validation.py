"""
Validation metrics for the surrogate GLM.

All metrics compare the GBM pseudo-predictions to the fitted GLM predictions.
The goal is to quantify how much discriminatory power and accuracy the GLM
retains relative to the GBM it is distilling.

Gini coefficient
    We use the weighted Gini coefficient (also called the concentration index)
    with exposure as weights. This is the standard discrimination metric in
    UK motor insurance. A GLM that retains 95%+ of the GBM's Gini is generally
    considered a faithful surrogate.

Deviance ratio
    Analogous to R-squared for GLMs. Measures how well the GLM's predictions
    explain the variance in the GBM pseudo-predictions. We use the appropriate
    deviance for the chosen family (Poisson or Gamma).

Segment deviation
    The maximum and mean absolute relative deviation between GBM and GLM
    predictions, computed at the level of each unique segment (combination of
    binned levels). This is the most operationally relevant metric: if the
    GLM is within 5% of the GBM in every cell, the factor tables are faithful.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def compute_gini(
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Compute the weighted Gini coefficient of ``predictions``.

    The Gini coefficient is defined as twice the area between the Lorenz
    curve and the diagonal. Values range from 0 (no discrimination) to 1
    (perfect discrimination).

    Parameters
    ----------
    predictions :
        1-D array of model predictions (pure, not normalised).
    weights :
        Optional 1-D array of exposure weights. If None, all observations
        have unit weight.

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    n = len(predictions)
    if n == 0:
        return 0.0

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    # Sort by predictions ascending
    order = np.argsort(predictions)
    pred_sorted = predictions[order]
    w_sorted = w[order]

    # Normalise weights
    w_total = w_sorted.sum()
    if w_total == 0:
        return 0.0

    w_norm = w_sorted / w_total
    pred_w_norm = (pred_sorted * w_sorted) / (pred_sorted * w_sorted).sum()

    # Lorenz curve: cumulative weighted prediction vs cumulative weight
    cum_w = np.cumsum(w_norm)
    cum_pred = np.cumsum(pred_w_norm)

    # Area under Lorenz curve via trapezoid rule
    # prepend (0, 0)
    cum_w = np.concatenate([[0.0], cum_w])
    cum_pred = np.concatenate([[0.0], cum_pred])
    area_lorenz = np.trapz(cum_pred, cum_w)

    gini = 1.0 - 2.0 * area_lorenz
    return float(np.clip(gini, 0.0, 1.0))


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Unit Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))."""
    mask = y > 0
    d = np.zeros_like(y)
    d[mask] = y[mask] * np.log(y[mask] / mu[mask]) - (y[mask] - mu[mask])
    d[~mask] = mu[~mask]  # y=0 term: 0 - (0 - mu) = mu
    return float(2.0 * d.sum())


def _gamma_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Unit Gamma deviance: 2 * sum(-log(y/mu) + (y - mu)/mu)."""
    ratio = y / mu
    d = -np.log(ratio) + (ratio - 1.0)
    return float(2.0 * d.sum())


def compute_deviance_ratio(
    pseudo: np.ndarray,
    glm_pred: np.ndarray,
    family: str,
) -> float:
    """
    Compute the deviance ratio (pseudo-R-squared) for the surrogate GLM.

    The null model uses the exposure-weighted mean of the pseudo-predictions.

    Parameters
    ----------
    pseudo :
        GBM pseudo-predictions (used as the response variable in the GLM).
    glm_pred :
        Fitted GLM predictions.
    family :
        Distribution family: ``"poisson"`` or ``"gamma"``.

    Returns
    -------
    float
        Deviance ratio in [0, 1]. Values above 0.90 are generally good.
    """
    pseudo = np.clip(pseudo, 1e-10, None)
    glm_pred = np.clip(glm_pred, 1e-10, None)

    null_mean = pseudo.mean()
    null_pred = np.full_like(pseudo, null_mean)

    if family == "poisson":
        dev_null = _poisson_deviance(pseudo, null_pred)
        dev_resid = _poisson_deviance(pseudo, glm_pred)
    elif family in ("gamma", "tweedie"):
        dev_null = _gamma_deviance(pseudo, null_pred)
        dev_resid = _gamma_deviance(pseudo, glm_pred)
    else:
        raise ValueError(f"Unknown family: {family!r}")

    if dev_null == 0:
        return 0.0

    return float(1.0 - dev_resid / dev_null)


def compute_segment_deviation(
    X_binned: pl.DataFrame,
    pseudo: np.ndarray,
    glm_pred: np.ndarray,
    exposure: np.ndarray,
    bin_features: list[str],
    cat_features: list[str],
) -> tuple[float, float, int]:
    """
    Compute max and mean absolute relative deviation between GBM and GLM,
    aggregated at the level of each unique segment.

    A segment is defined by the unique combination of all binned and
    categorical feature levels. Within each segment we compute the
    exposure-weighted average of the GBM predictions and GLM predictions,
    then measure their relative deviation.

    Parameters
    ----------
    X_binned :
        DataFrame with ``__bin`` columns (output of OptimalBinner.transform).
    pseudo :
        GBM pseudo-predictions.
    glm_pred :
        GLM predictions.
    exposure :
        Exposure weights.
    bin_features :
        List of ``<feature>__bin`` column names to use as segment keys.
    cat_features :
        List of raw categorical column names to use as segment keys.

    Returns
    -------
    max_deviation : float
        Maximum absolute relative deviation across segments.
    mean_deviation : float
        Exposure-weighted mean absolute relative deviation.
    n_segments : int
        Number of distinct segments.
    """
    group_cols = [c for c in bin_features if c in X_binned.columns]
    group_cols += [c for c in cat_features if c in X_binned.columns]

    if not group_cols:
        # No grouping possible - compute global deviation
        rel_dev = abs(pseudo.mean() - glm_pred.mean()) / max(pseudo.mean(), 1e-10)
        return float(rel_dev), float(rel_dev), 1

    df = X_binned.select(group_cols).with_columns(
        [
            pl.Series("__pseudo", pseudo),
            pl.Series("__glm", glm_pred),
            pl.Series("__exposure", exposure),
        ]
    )

    agg = df.group_by(group_cols).agg(
        [
            (pl.col("__pseudo") * pl.col("__exposure")).sum().alias("pseudo_weighted"),
            (pl.col("__glm") * pl.col("__exposure")).sum().alias("glm_weighted"),
            pl.col("__exposure").sum().alias("exposure_sum"),
        ]
    )

    agg = agg.with_columns(
        [
            (pl.col("pseudo_weighted") / pl.col("exposure_sum")).alias("avg_pseudo"),
            (pl.col("glm_weighted") / pl.col("exposure_sum")).alias("avg_glm"),
        ]
    ).with_columns(
        (
            (pl.col("avg_glm") - pl.col("avg_pseudo")).abs()
            / pl.col("avg_pseudo").clip(lower_bound=1e-10)
        ).alias("rel_deviation")
    )

    max_dev = float(agg["rel_deviation"].max())
    # Exposure-weighted mean
    total_exposure = float(agg["exposure_sum"].sum())
    mean_dev = float(
        (agg["rel_deviation"] * agg["exposure_sum"]).sum() / max(total_exposure, 1e-10)
    )
    n_seg = len(agg)

    return max_dev, mean_dev, n_seg


def double_lift_chart(
    pseudo: np.ndarray,
    glm_pred: np.ndarray,
    exposure: np.ndarray | None = None,
    n_deciles: int = 10,
) -> pl.DataFrame:
    """
    Compute a double-lift chart comparing GBM and GLM predictions.

    Rows are sorted by the ratio of GBM prediction to GLM prediction,
    then grouped into deciles. The chart shows whether the GBM and GLM
    agree on the ranking of risks.

    Parameters
    ----------
    pseudo :
        GBM pseudo-predictions.
    glm_pred :
        GLM predictions.
    exposure :
        Optional exposure weights.
    n_deciles :
        Number of bands (default 10).

    Returns
    -------
    pl.DataFrame
        Columns: decile, avg_gbm, avg_glm, ratio_gbm_glm, exposure_share.
    """
    w = np.ones_like(pseudo) if exposure is None else np.asarray(exposure, dtype=float)

    ratio = pseudo / np.clip(glm_pred, 1e-10, None)
    order = np.argsort(ratio)

    pseudo_sorted = pseudo[order]
    glm_sorted = glm_pred[order]
    w_sorted = w[order]

    # Assign deciles based on cumulative weight
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    decile_idx = np.minimum(
        (n_deciles * cum_w / total_w).astype(int), n_deciles - 1
    )

    rows = []
    for d in range(n_deciles):
        mask = decile_idx == d
        if not mask.any():
            continue
        w_d = w_sorted[mask]
        pseudo_d = pseudo_sorted[mask]
        glm_d = glm_sorted[mask]
        total_d = w_d.sum()

        avg_gbm = float((pseudo_d * w_d).sum() / max(total_d, 1e-10))
        avg_glm = float((glm_d * w_d).sum() / max(total_d, 1e-10))
        rows.append(
            {
                "decile": d + 1,
                "avg_gbm": avg_gbm,
                "avg_glm": avg_glm,
                "ratio_gbm_to_glm": avg_gbm / max(avg_glm, 1e-10),
                "exposure_share": float(total_d / max(total_w, 1e-10)),
            }
        )

    return pl.DataFrame(rows)
