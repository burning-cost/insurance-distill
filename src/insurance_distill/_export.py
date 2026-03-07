"""
Export GLM factor tables from the fitted surrogate model.

Factor tables are the primary deliverable of this library. Each table
describes, for one variable, the multiplicative rating factor for each
level relative to the base level.

The convention used here matches Radar and Emblem:
- The base level of each variable has a relativity of exactly 1.0.
- The base level is chosen to be the first bin (lexicographic order after
  the reference coding applied during fitting). This is consistent with the
  GLM's reference-coding convention: the first level is the intercept absorber.
- Relativities are on the multiplicative (natural) scale: exp(coefficient).

Output columns per factor table:

    level           : str  - bin label or category value
    log_coefficient : float - raw GLM log-scale coefficient (0.0 for base)
    relativity      : float - exp(log_coefficient), multiplicative factor
    n_obs           : int   - number of training rows in this level (informational)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ._types import BinSpec


def build_factor_tables(
    glm: Any,
    bin_specs: dict[str, BinSpec],
    col_names: list[str],
    intercept: float,
    cat_features: list[str],
) -> dict[str, pl.DataFrame]:
    """
    Build one factor table DataFrame per feature from fitted GLM coefficients.

    Parameters
    ----------
    glm :
        Fitted glum GeneralizedLinearRegressor.
    bin_specs :
        Mapping of feature name to its BinSpec (from OptimalBinner).
    col_names :
        Design matrix column names in the same order as glm.coef_.
        Each name is of the form ``"<feature>__bin=<level>"`` or
        ``"<cat_feature>=<level>"``.
    intercept :
        GLM intercept (log scale).
    cat_features :
        List of raw categorical feature names (no ``__bin`` suffix).

    Returns
    -------
    dict[str, pl.DataFrame]
        One DataFrame per feature.
    """
    coef = np.asarray(glm.coef_, dtype=float)
    coef_map: dict[str, float] = dict(zip(col_names, coef))

    tables: dict[str, pl.DataFrame] = {}

    # Continuous features (have bin specs)
    for feat, spec in bin_specs.items():
        col_prefix = f"{feat}__bin="
        rows = []
        for label in spec.bin_labels:
            col_key = f"{col_prefix}{label}"
            log_coef = coef_map.get(col_key, 0.0)  # 0.0 = reference level
            rows.append(
                {
                    "level": label,
                    "log_coefficient": log_coef,
                    "relativity": float(np.exp(log_coef)),
                }
            )
        tables[feat] = pl.DataFrame(rows)

    # Categorical features
    for feat in cat_features:
        col_prefix = f"{feat}="
        # Find all levels for this feature in the coefficient map
        matching = {
            k: v for k, v in coef_map.items() if k.startswith(col_prefix)
        }
        # Also need the reference level (coefficient = 0)
        all_levels_encoded = sorted(matching.keys())

        # Reconstruct base level name from the coefficient map
        # The base level is the one that was dropped (reference coding)
        # We can infer it exists but has no entry in coef_map
        rows = []
        # Base level placeholder - we do not know its name without the original data
        # so we flag it as "(base)"
        rows.append(
            {
                "level": "(base - reference level)",
                "log_coefficient": 0.0,
                "relativity": 1.0,
            }
        )
        for col_key in all_levels_encoded:
            level_name = col_key[len(col_prefix):]
            log_coef = matching[col_key]
            rows.append(
                {
                    "level": level_name,
                    "log_coefficient": log_coef,
                    "relativity": float(np.exp(log_coef)),
                }
            )
        tables[feat] = pl.DataFrame(rows)

    return tables


def build_glm_coefficients_df(
    glm: Any,
    col_names: list[str],
    intercept: float,
) -> pl.DataFrame:
    """
    Build a tidy DataFrame of all GLM coefficients.

    Parameters
    ----------
    glm :
        Fitted glum GeneralizedLinearRegressor.
    col_names :
        Design matrix column names (same order as glm.coef_).
    intercept :
        Log-scale intercept.

    Returns
    -------
    pl.DataFrame
        Columns: term, log_coefficient, relativity.
    """
    coef = np.asarray(glm.coef_, dtype=float)

    rows = [
        {
            "term": "(Intercept)",
            "log_coefficient": intercept,
            "relativity": float(np.exp(intercept)),
        }
    ]
    for name, c in zip(col_names, coef):
        rows.append(
            {
                "term": name,
                "log_coefficient": float(c),
                "relativity": float(np.exp(c)),
            }
        )

    return pl.DataFrame(rows)


def format_radar_csv(factor_table: pl.DataFrame, feature_name: str) -> str:
    """
    Format a factor table as a Radar-compatible CSV string.

    Radar expects a simple two-column format:

        FeatureName,Relativity
        [lo, hi),1.000
        ...

    Parameters
    ----------
    factor_table :
        Factor table DataFrame from :func:`build_factor_tables`.
    feature_name :
        Name to write in the header row.

    Returns
    -------
    str
        CSV content as a string (write with open(path, "w").write(...)).
    """
    lines = [f"{feature_name},Relativity"]
    for row in factor_table.iter_rows(named=True):
        level = row["level"]
        rel = row["relativity"]
        lines.append(f"{level},{rel:.6f}")
    return "\n".join(lines) + "\n"
