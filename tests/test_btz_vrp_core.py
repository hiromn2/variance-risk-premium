"""Offline unit tests for btz_vrp_replication_validated.py.

Run from the project root with:
    pytest tests/test_btz_vrp_core.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "btz_vrp_replication_validated.py"
spec = importlib.util.spec_from_file_location("btz", MODULE_PATH)
btz = importlib.util.module_from_spec(spec)
sys.modules["btz"] = btz
assert spec.loader is not None
spec.loader.exec_module(btz)


def test_monthly_rv_units():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)

    rv = btz.build_monthly_rv(returns)

    expected = 12.0 * (0.01**2 + (-0.02)**2 + 0.03**2)
    assert np.isclose(rv.loc[pd.Timestamp("2020-01-31")], expected)


def test_vix2_units():
    dates = pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-28"])
    vix = pd.Series([15.0, 20.0, 30.0], index=dates)

    vix2 = btz.build_monthly_vix2(vix)

    assert np.isclose(vix2.loc[pd.Timestamp("2020-01-31")], 0.20**2)
    assert np.isclose(vix2.loc[pd.Timestamp("2020-02-29")], 0.30**2)


def test_forward_return_alignment():
    # One synthetic daily return per month; monthly return equals that value.
    dates = pd.to_datetime([
        "2020-01-15", "2020-02-15", "2020-03-15",
        "2020-04-15", "2020-05-15", "2020-06-15",
    ])
    monthly_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], index=dates)

    month_ends = pd.date_range("2020-01-31", periods=6, freq="ME")
    vix2 = pd.Series(0.04, index=month_ends, name="VIX2")
    rv = pd.Series(0.02, index=month_ends, name="RV")
    evt_rv = pd.Series(0.018, index=month_ends, name="EVT_RV")

    panel = btz.build_panel(vix2, rv, evt_rv, monthly_returns, macro=None, cape=None)

    # At Jan 2020, ret_3m should be Feb + Mar + Apr, not Jan + Feb + Mar.
    assert np.isclose(panel.loc[pd.Timestamp("2020-01-31"), "ret_3m"], 0.02 + 0.03 + 0.04)


def test_gpd_conditional_second_moment_exponential_case():
    # xi=0 => exponential exceedance y with E[y]=beta and E[y^2]=2 beta^2.
    xi, beta, u = 0.0, 2.0, 1.0
    moment = btz.gpd_conditional_second_moment(xi, beta, u)
    expected = u**2 + 2 * u * beta + 2 * beta**2
    assert np.isclose(moment, expected)


def test_gpd_conditional_second_moment_invalid_xi_fallback():
    assert np.isnan(btz.gpd_conditional_second_moment(0.5, 1.0, 0.01))
    assert np.isnan(btz.gpd_conditional_second_moment(0.7, 1.0, 0.01))


def test_oos_r2_no_future_leakage_for_h3():
    n = 20
    index = pd.date_range("2000-01-31", periods=n, freq="ME")
    panel = pd.DataFrame({
        "VRP": np.linspace(-0.02, 0.02, n),
        "ret_3m": np.linspace(0.01, 0.10, n),
    }, index=index)

    out = btz.oos_r2(panel, vrp_col="VRP", h=3, min_train=5, return_predictions=True)
    preds = out["predictions"].dropna(subset=["pred_model"])

    # First h=3 forecast with min_train=5 should train on exactly 5 observations.
    assert preds["train_end_obs"].iloc[0] == 5

    # For every forecast at positional index t, training endpoint must be t-h+1.
    for timestamp, row in preds.iterrows():
        t = panel.index.get_loc(timestamp)
        assert row["train_end_obs"] == t - 3 + 1


def test_rolling_oos_r2_worse_post_2008():
    """Rolling OOS R² is closer to zero (less negative) pre-2008 than post-2008.

    Uses the cached panel so the test reflects the real data distribution.
    The structural-break story requires pre-2008 OOS R² > post-2008 OOS R².
    """
    import pytest
    parquet = Path(__file__).resolve().parents[1] / "data" / "processed" / "panel_us.parquet"
    if not parquet.exists():
        pytest.skip("panel_us.parquet not found; run the main pipeline first")

    panel = pd.read_parquet(parquet).sort_index()
    pre = panel.loc[:"2007-12-31"]
    post = panel.loc["2008-01-01":]

    r2_pre = btz.rolling_oos_r2(pre, vrp_col="VRP", h=1, window=60)
    r2_post = btz.rolling_oos_r2(post, vrp_col="VRP", h=1, window=60)

    assert np.isfinite(r2_pre["OOS_R2_full"]), "pre-2008 OOS R² is NaN — not enough data"
    assert np.isfinite(r2_post["OOS_R2_full"]), "post-2008 OOS R² is NaN — not enough data"
    assert r2_pre["OOS_R2_full"] > r2_post["OOS_R2_full"], (
        f"Expected pre-2008 OOS R² ({r2_pre['OOS_R2_full']:.4f}) > "
        f"post-2008 ({r2_post['OOS_R2_full']:.4f})"
    )


def test_gpd_second_moment_monotone_in_xi():
    """Larger ξ (heavier tail) implies larger conditional second moment, β and u fixed.

    Validates theoretical monotonicity: as ξ ↑ toward 0.5, both E[y] and E[y²]
    increase, so the EVT correction assigns more variance to tail days in
    high-ξ regimes — which is the intended behavior of the tail correction.
    """
    beta, u = 0.01, 0.02
    xis = [0.0, 0.1, 0.2, 0.3, 0.4]
    moments = [btz.gpd_conditional_second_moment(xi, beta, u) for xi in xis]

    assert all(np.isfinite(m) for m in moments), "Some moments are NaN for valid ξ < 0.5"
    assert all(m1 < m2 for m1, m2 in zip(moments, moments[1:])), (
        f"Moments not strictly increasing with ξ: {list(zip(xis, moments))}"
    )
