"""Tests for _export.py - factor table building and CSV formatting."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_distill._export import (
    build_glm_coefficients_df,
    format_radar_csv,
)


def test_glm_coefficients_df_includes_intercept():
    """Check intercept row is always first."""

    class FakeGLM:
        coef_ = np.array([0.1, -0.2, 0.05])
        intercept_ = -3.0

    df = build_glm_coefficients_df(
        glm=FakeGLM(),
        col_names=["age__bin=[18.00, 25.00)", "age__bin=[25.00, 40.00)", "region=North"],
        intercept=-3.0,
    )
    assert df["term"][0] == "(Intercept)"
    assert df["relativity"][0] == pytest.approx(np.exp(-3.0), rel=1e-6)
    assert len(df) == 4  # 1 intercept + 3 coefficients


def test_glm_coefficients_df_relativities_positive():
    class FakeGLM:
        coef_ = np.array([0.5, -0.3])
        intercept_ = -2.0

    df = build_glm_coefficients_df(
        glm=FakeGLM(),
        col_names=["x=a", "x=b"],
        intercept=-2.0,
    )
    assert (df["relativity"].to_numpy() > 0).all()


def test_format_radar_csv():
    df = pl.DataFrame(
        {
            "level": ["[18.00, 25.00)", "[25.00, 40.00)", "[40.00, +inf)"],
            "log_coefficient": [0.3, 0.0, -0.1],
            "relativity": [1.35, 1.0, 0.905],
        }
    )
    csv_str = format_radar_csv(df, "driver_age")
    lines = csv_str.strip().split("\n")
    assert lines[0] == "driver_age,Relativity"
    assert len(lines) == 4  # header + 3 data rows
    # Check float formatting
    assert "1.350000" in lines[1]
