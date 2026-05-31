"""
Variance Risk Premium Replication & EVT Extension
Based on: Bollerslev, Tauchen & Zhou (2009), RFS
Extension: EVT-corrected realized variance, motivated by Londono (2011) puzzle

This version implements the validation pass requested before public polish:
  1. Explicit sample-period controls: BTZ 1990-2007, post-2008, and full sample.
  2. Corrected EVT tail replacement using E[|r|^2 | |r| > u].
  3. Robustness tables saved for each subsample before any large refactor.
  4. Output folders for tables and figures.
  5. Safer out-of-sample R² indexing for h-month overlapping returns.
  6. Optional data dependencies, so analytical functions can be unit-tested offline.

Usage:
  pip install yfinance fredapi scipy statsmodels matplotlib pandas numpy openpyxl xlrd python-dotenv
  export $(cat .env | xargs)  # expects FRED_API_KEY=...
  python btz_vrp_replication_validated.py

Useful variants:
  python btz_vrp_replication_validated.py --end 2026-05-29
  python btz_vrp_replication_validated.py --skip-plots
  python btz_vrp_replication_validated.py --run-australia
"""

# ============================================================
# 0. DEPENDENCIES AND CONFIG
# ============================================================
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # allows offline unit tests for analytical functions
    yf = None

try:
    from fredapi import Fred
except ImportError:  # allows offline unit tests for analytical functions
    Fred = None

from scipy.stats import genpareto
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; saves to file
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

DEFAULT_START = "1990-01-01"
# Use today's date by default so the full sample naturally extends to current available data.
DEFAULT_END = os.getenv("VRP_END", datetime.today().strftime("%Y-%m-%d"))
BTZ_START = "1990-01-01"
BTZ_END = "2007-12-31"
POST_START = "2008-01-01"

_ROOT = Path(__file__).resolve().parent
DATA_DIR = _ROOT / "data" / "processed"
FIGURE_DIR = _ROOT / "reports" / "figures"
TABLE_DIR = _ROOT / "reports" / "tables"


@dataclass(frozen=True)
class SampleSpec:
    label: str
    start: str
    end: str


def require_yfinance() -> None:
    if yf is None:
        raise ImportError(
            "yfinance is required for data download. Install with: pip install yfinance"
        )


def make_fred() -> Optional["Fred"]:
    """Return a Fred client if fredapi is installed; otherwise return None."""
    if Fred is None:
        return None
    return Fred(api_key=os.getenv("FRED_API_KEY"))


def ensure_output_dirs() -> None:
    for path in [DATA_DIR, FIGURE_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DATA ACQUISITION
# ============================================================

def fetch_vix(start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.Series:
    """
    CBOE VIX daily close from Yahoo Finance (^VIX).

    VIX is annualized volatility in percent. VIX2 = (VIX / 100)^2 is annualized
    variance in decimal^2 units.
    """
    require_yfinance()
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"].squeeze()
    vix.name = "VIX"
    return vix.dropna()


def fetch_spx_returns(start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.Series:
    """S&P 500 daily log-returns from Yahoo Finance (^GSPC)."""
    require_yfinance()
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"].squeeze()
    ret = np.log(spx / spx.shift(1)).dropna()
    ret.name = "spx_logret"
    return ret


def fetch_asx_returns(start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.Series:
    """ASX 200 daily log-returns (^AXJO). Yahoo coverage usually starts around 2000."""
    require_yfinance()
    asx = yf.download("^AXJO", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"].squeeze()
    ret = np.log(asx / asx.shift(1)).dropna()
    ret.name = "asx_logret"
    return ret


def fetch_avix(start: str = DEFAULT_START, end: str = DEFAULT_END) -> Optional[pd.Series]:
    """
    ASX 200 VIX. yfinance usually has sparse/no ^AVIX coverage.

    Keep this optional. The main validation pass should not depend on Australia.
    """
    require_yfinance()
    try:
        avix = yf.download("^AVIX", start=start, end=end, auto_adjust=True,
                           progress=False)["Close"].squeeze()
        avix = avix.dropna()
        if avix.empty or len(avix) < 100:
            print("WARNING: ^AVIX data sparse. Australia VRP requires manual ASX data.")
            return None
        avix.name = "AVIX"
        return avix
    except Exception as exc:
        print(f"AVIX fetch failed: {exc}")
        return None


def fetch_macro_controls(start: str = DEFAULT_START, end: str = DEFAULT_END) -> Optional[pd.DataFrame]:
    """
    FRED macro controls used in BTZ-style regressions.

    Required derived controls:
      term_spread    = GS10 - TB3MS
      default_spread = BAA - AAA
    """
    fred = make_fred()
    if fred is None:
        print("WARNING: fredapi not installed. Skipping macro controls.")
        return None

    series = {
        "gs10": "GS10",
        "tb3m": "TB3MS",
        "baa": "BAA",
        "aaa": "AAA",
    }
    frames: dict[str, pd.Series] = {}
    for name, fred_id in series.items():
        try:
            s = fred.get_series(fred_id, observation_start=start, observation_end=end)
            s.name = name
            frames[name] = s
        except Exception as exc:
            print(f"FRED fetch failed for {fred_id}: {exc}")

    if not frames:
        return None

    macro = pd.DataFrame(frames)
    if {"gs10", "tb3m"}.issubset(macro.columns):
        macro["term_spread"] = macro["gs10"] - macro["tb3m"]
    if {"baa", "aaa"}.issubset(macro.columns):
        macro["default_spread"] = macro["baa"] - macro["aaa"]
    return macro


def fetch_cape(start: str = DEFAULT_START, end: str = DEFAULT_END) -> Optional[pd.Series]:
    """
    Shiller CAPE from Robert Shiller's Excel file.

    We use log(P / E10), with P and E coerced to numeric before rolling.
    """
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    try:
        df = pd.read_excel(url, sheet_name="Data", header=7)
        df = df[["Date", "P", "E"]].copy()
        df = df[df["Date"].notna() & df["P"].notna()].copy()

        # Date column is decimal year/month, e.g. 1990.01 = Jan 1990.
        df["Date"] = pd.to_datetime(
            df["Date"].astype(str).str[:7],
            format="%Y.%m",
            errors="coerce",
        )
        df = df.dropna(subset=["Date"]).set_index("Date")
        df["P"] = pd.to_numeric(df["P"], errors="coerce")
        df["E"] = pd.to_numeric(df["E"], errors="coerce")
        df["E10"] = df["E"].rolling(120, min_periods=120).mean()
        df["cape"] = np.log(df["P"] / df["E10"])
        cape = df["cape"].dropna()
        cape.index = cape.index.to_period("M").to_timestamp("M")
        # Shiller file sometimes has duplicated or non-monotonic dates after period conversion.
        cape = cape[~cape.index.duplicated(keep="last")]
        cape = cape.sort_index()
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        cape = cape[(cape.index >= start_dt) & (cape.index <= end_dt)]
        cape.name = "log_cape"
        return cape
    except Exception as exc:
        print(f"CAPE fetch failed: {exc}. Skipping CAPE control.")
        return None


# ============================================================
# 2. REALIZED VARIANCE CONSTRUCTION
# ============================================================

def build_monthly_rv(daily_returns: pd.Series) -> pd.Series:
    """
    Standard monthly realized variance: 12 * sum of squared daily log-returns.

    Units: annualized variance, decimal^2.
    """
    monthly = (
        daily_returns
        .groupby(daily_returns.index.to_period("M"))
        .apply(lambda x: float((x ** 2).sum() * 12))
    )
    monthly.index = monthly.index.to_timestamp("M")
    monthly.name = "RV"
    return monthly


def build_monthly_vix2(daily_vix: pd.Series) -> pd.Series:
    """
    Month-end VIX-implied annualized variance: (VIX / 100)^2.
    """
    monthly = daily_vix.groupby(daily_vix.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp("M")
    vix2 = (monthly / 100.0) ** 2
    vix2.name = "VIX2"
    return vix2


# ============================================================
# 3. EVT-CORRECTED REALIZED VARIANCE
# ============================================================

def fit_gpd_tail(returns: np.ndarray, threshold_quantile: float = 0.90) -> tuple[float, float, float]:
    """
    Fit a GPD to exceedances of |r| over threshold u.

    Returns:
      xi   : GPD shape parameter
      beta : GPD scale parameter
      u    : absolute-return threshold
    """
    abs_r = np.abs(np.asarray(returns, dtype=float))
    abs_r = abs_r[np.isfinite(abs_r)]
    if len(abs_r) == 0:
        return 0.0, np.nan, np.nan

    # Threshold choice: threshold_quantile sets the GPD fitting boundary u.
    #
    # Why q85 (0.85) fails or flips sign:
    #   At the 85th percentile the exceedance set includes a large fraction of
    #   the distribution body, not just extreme tail events. GPD asymptotics
    #   (Pickands–Balkema–de Haan theorem) are valid only for genuine tail data.
    #   Fitting GPD to body observations biases ξ downward — sometimes negative —
    #   which corrupts the second-moment correction and can flip the sign of
    #   delta-VRP in the horse-race regression.
    #
    # Why the 252-day window dominates:
    #   One trading year is the natural regime-adaptation horizon. Longer windows
    #   (504, 756 days) average across structural breaks, diluting the current
    #   tail shape estimate with stale observations from earlier regimes. This
    #   smooths out ξ and weakens the EVT correction.
    #
    # Minimum-exceedances guard: a 252-day window at q90 produces ~25 exceedances
    #   (252 × 0.10 = 25.2). The guard must be below that; 15 is the lower bound
    #   for stable GPD MLE in finite samples. The original value of 30 was written
    #   for full-sample estimation and silently triggered the fallback on every
    #   rolling window call, storing ξ = 0 for the entire sample.
    u = float(np.quantile(abs_r, threshold_quantile))
    exceedances = abs_r[abs_r > u] - u

    if len(exceedances) < 15:
        # Too few exceedances for reliable MLE; exponential-like fallback.
        beta = float(np.std(abs_r))
        return 0.0, beta, u

    xi, loc, beta = genpareto.fit(exceedances, floc=0)
    return float(xi), float(beta), u


def gpd_conditional_second_moment(xi: float, beta: float, u: float) -> float:
    """
    Compute E[|r|^2 | |r| > u] where |r| = u + y and y ~ GPD(xi, beta).

    GPD moments:
      E[y]   = beta / (1 - xi),       valid for xi < 1
      E[y^2] = 2 beta^2 / ((1-xi)(1-2xi)), valid for xi < 1/2

    This is the corrected object needed when replacing each observed tail-day
    squared return by its GPD conditional expectation.
    """
    if not np.isfinite([xi, beta, u]).all() or beta <= 0 or xi >= 0.5:
        return np.nan

    ey = beta / (1.0 - xi)
    ey2 = 2.0 * beta ** 2 / ((1.0 - xi) * (1.0 - 2.0 * xi))
    return float(u ** 2 + 2.0 * u * ey + ey2)


def build_evt_rv(
    daily_returns: pd.Series,
    window: int = 252,
    threshold_quantile: float = 0.90,
    return_xi: bool = False,
) -> "pd.Series | tuple[pd.Series, pd.Series]":
    """
    Build EVT-corrected monthly realized variance.

    For each month t:
      1. Estimate GPD on trailing `window` daily returns ending before month-end.
      2. Compute body contribution directly: sum r^2 for |r| <= u.
      3. Replace each observed tail-day r^2 by E[|r|^2 | |r| > u].
      4. Annualize by multiplying by 12.

    If return_xi=True, returns (evt_rv, xi_series) where xi_series holds the
    monthly GPD shape parameter (ξ) estimates for rolling diagnostic plots.
    """
    daily_returns = daily_returns.dropna().sort_index()
    rets = daily_returns.values
    dates = daily_returns.index
    monthly_periods = daily_returns.groupby(daily_returns.index.to_period("M"))

    evt_rv_values: list[float] = []
    evt_rv_dates: list[pd.Timestamp] = []
    xi_values: list[float] = []

    for period, group in monthly_periods:
        month_end_idx = dates.get_loc(group.index[-1])
        if month_end_idx < window:
            continue

        # Avoid future leakage: estimate GPD on observations strictly before this month-end.
        train = rets[month_end_idx - window: month_end_idx]
        xi, beta, u = fit_gpd_tail(train, threshold_quantile)
        tail_second_moment = gpd_conditional_second_moment(xi, beta, u)

        r_month = group.values
        abs_r = np.abs(r_month)
        tail_mask = abs_r > u
        body_rv = float(np.sum(r_month[~tail_mask] ** 2))

        if np.isnan(tail_second_moment):
            tail_rv = float(np.sum(r_month[tail_mask] ** 2))
        else:
            tail_rv = int(np.sum(tail_mask)) * tail_second_moment

        evt_rv_values.append((body_rv + tail_rv) * 12.0)
        evt_rv_dates.append(period.to_timestamp("M"))
        xi_values.append(xi)

    evt_rv = pd.Series(evt_rv_values, index=pd.DatetimeIndex(evt_rv_dates), name="EVT_RV")
    if return_xi:
        xi_series = pd.Series(xi_values, index=pd.DatetimeIndex(evt_rv_dates), name="gpd_xi")
        return evt_rv, xi_series
    return evt_rv


# Backwards-compatible alias for older notebooks that imported evt_tail_variance.
def evt_tail_variance(xi: float, beta: float, u: float, n_obs: int | None = None,
                      n_exceed: int | None = None) -> float:
    """
    Deprecated compatibility wrapper.

    Use gpd_conditional_second_moment(xi, beta, u) instead. n_obs and n_exceed
    are ignored because the tail replacement is conditional on observing a tail day.
    """
    return gpd_conditional_second_moment(xi, beta, u)


# ============================================================
# 4. PANEL CONSTRUCTION
# ============================================================

def build_panel(
    vix2: pd.Series,
    rv: pd.Series,
    evt_rv: Optional[pd.Series],
    spx_ret: pd.Series,
    macro: Optional[pd.DataFrame],
    cape: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Merge monthly series and construct forward returns.

    ret_hm at month t equals the sum of returns from t+1 through t+h.
    """
    monthly_ret = spx_ret.groupby(spx_ret.index.to_period("M")).sum()
    monthly_ret.index = monthly_ret.index.to_timestamp("M")
    monthly_ret.name = "ret_1m"

    series_to_concat = [vix2, rv, monthly_ret]
    if evt_rv is not None:
        series_to_concat.append(evt_rv)
    panel = pd.concat(series_to_concat, axis=1).dropna(subset=["VIX2", "RV"])

    panel["VRP"] = panel["VIX2"] - panel["RV"]
    if "EVT_RV" in panel.columns:
        panel["EVT_VRP"] = panel["VIX2"] - panel["EVT_RV"]

    if macro is not None and not macro.empty:
        macro_m = macro.resample("ME").last()
        join_cols = [c for c in ["term_spread", "default_spread", "tb3m"] if c in macro_m.columns]
        if join_cols:
            panel = panel.join(macro_m[join_cols], how="left")

    if cape is not None:
        panel = panel.join(cape.rename("log_cape"), how="left")
        panel["log_cape"] = panel["log_cape"].ffill()

    # Forward returns: at t, forecast cumulative return t+1...t+h.
    for h in [1, 3, 6, 12]:
        panel[f"ret_{h}m"] = monthly_ret.rolling(h).sum().shift(-h)

    if "tb3m" in panel.columns:
        rf = panel["tb3m"] / 100.0 / 12.0
        for h in [1, 3, 6, 12]:
            panel[f"xret_{h}m"] = panel[f"ret_{h}m"] - rf * h

    return panel.dropna(subset=["VRP"])


def slice_panel(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice a monthly panel by calendar dates."""
    return panel.loc[pd.to_datetime(start):pd.to_datetime(end)].copy()


# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================

def summary_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, skewness, excess kurtosis, AR(1), min, max, and N."""
    cols = [c for c in ["VRP", "RV", "VIX2", "EVT_RV", "EVT_VRP", "log_cape"] if c in panel.columns]
    stats: dict[str, dict[str, float]] = {}
    for c in cols:
        s = panel[c].dropna()
        stats[c] = {
            "mean": s.mean(),
            "std": s.std(),
            "skew": s.skew(),
            "kurt": s.kurtosis(),
            "AR(1)": s.autocorr(lag=1),
            "min": s.min(),
            "max": s.max(),
            "N": len(s),
        }
    return pd.DataFrame(stats).T.round(6)


# ============================================================
# 6. HODRICK (1992) STANDARD ERRORS
# ============================================================

def hodrick_se(y: np.ndarray, X: np.ndarray, h: int) -> np.ndarray:
    """Hodrick-style covariance estimator for overlapping h-period returns."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    if T <= k:
        return np.full(k, np.nan)

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta

    XtX_inv = np.linalg.pinv(X.T @ X / T)

    S = np.zeros((k, k))
    for t in range(T):
        S += np.outer(e[t] * X[t], e[t] * X[t])
    S /= T

    for j in range(1, max(int(h), 1)):
        G = np.zeros((k, k))
        for t in range(j, T):
            G += np.outer(e[t] * X[t], e[t - j] * X[t - j])
        G /= T
        S += G + G.T

    V = XtX_inv @ S @ XtX_inv / T
    diag = np.diag(V)
    diag = np.where((diag > 0) & np.isfinite(diag), diag, np.nan)
    return np.sqrt(diag)


def hac_se(y: np.ndarray, X: np.ndarray, h: int) -> np.ndarray:
    """Newey-West / HAC standard errors with maxlags = h − 1."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    lags = max(int(h) - 1, 0)
    try:
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return np.asarray(model.bse, dtype=float)
    except Exception:
        return np.full(X.shape[1], np.nan)


def safe_tstats(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    return np.divide(
        beta, se,
        out=np.full_like(beta, np.nan, dtype=float),
        where=(se > 0) & np.isfinite(se),
    )


# ============================================================
# 7. PREDICTABILITY REGRESSIONS
# ============================================================

def vrp_predictability(
    panel: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 6, 12),
    vrp_col: str = "VRP",
    controls: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """OLS return-predictability regressions with Hodrick SEs."""
    results: list[dict[str, object]] = []
    ctrl_cols = list(controls or [])

    for h in horizons:
        dep = f"xret_{h}m" if f"xret_{h}m" in panel.columns else f"ret_{h}m"
        reg_cols = [vrp_col] + [c for c in ctrl_cols if c in panel.columns]
        df = panel[[dep] + reg_cols].dropna()
        if len(df) < 50 or vrp_col not in reg_cols:
            continue

        y = df[dep].values
        X = sm.add_constant(df[reg_cols].values)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        se = hodrick_se(y, X, h)
        t_stats = safe_tstats(beta_hat, se)

        y_hat = X @ beta_hat
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        results.append({
            "horizon": h,
            "beta_VRP": beta_hat[1],
            "se_VRP": se[1],
            "t_VRP": t_stats[1],
            "R2": r2,
            "N": len(df),
            "controls": ", ".join([c for c in ctrl_cols if c in panel.columns]) or "none",
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).set_index("horizon")


# ============================================================
# 8. OUT-OF-SAMPLE R²
# ============================================================

def oos_r2(
    panel: pd.DataFrame,
    vrp_col: str = "VRP",
    h: int = 1,
    min_train: int = 60,
    return_predictions: bool = True,
) -> dict[str, object]:
    """
    Campbell-Thompson out-of-sample R² with leakage-safe expanding windows.

    For h-month forward returns, y_j is only known by forecast origin t when
    j + h <= t. Therefore, when forecasting y_t, the training set ends at
    train_end = t - h + 1 in Python-exclusive indexing.
    """
    dep = f"xret_{h}m" if f"xret_{h}m" in panel.columns else f"ret_{h}m"
    df = panel[[vrp_col, dep]].dropna()
    T = len(df)
    y = df[dep].values
    x = df[vrp_col].values

    pred_model = np.full(T, np.nan)
    pred_bench = np.full(T, np.nan)
    train_end_obs = np.full(T, np.nan)

    first_forecast = min_train + h - 1
    for t in range(first_forecast, T):
        train_end = t - h + 1  # exclusive endpoint; no future return leakage
        if train_end < min_train:
            continue

        y_train = y[:train_end]
        x_train = x[:train_end]
        pred_bench[t] = float(np.mean(y_train))
        train_end_obs[t] = train_end

        X_train = np.column_stack([np.ones(train_end), x_train])
        try:
            b = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            pred_model[t] = b[0] + b[1] * x[t]
        except Exception:
            pred_model[t] = pred_bench[t]

    valid = ~np.isnan(pred_model) & ~np.isnan(pred_bench)
    if valid.sum() == 0:
        result = {"OOS_R2": np.nan, "cum_dmse": pd.Series(dtype=float), "N_eval": 0}
    else:
        e_bench = (y[valid] - pred_bench[valid]) ** 2
        e_model = (y[valid] - pred_model[valid]) ** 2
        oos_r2_val = 1.0 - e_model.mean() / e_bench.mean()
        cum_dmse = pd.Series(np.cumsum(e_bench - e_model), index=df.index[valid])
        result = {"OOS_R2": oos_r2_val, "cum_dmse": cum_dmse, "N_eval": int(valid.sum())}

    if return_predictions:
        result["predictions"] = pd.DataFrame({
            "y": y,
            "pred_model": pred_model,
            "pred_bench": pred_bench,
            "train_end_obs": train_end_obs,
        }, index=df.index)
    return result


def rolling_oos_r2(
    panel: pd.DataFrame,
    vrp_col: str = "VRP",
    h: int = 1,
    window: int = 60,
) -> dict[str, object]:
    """
    Rolling-window OOS R² with a fixed estimation window.

    Uses a fixed `window`-month lookback rather than an expanding window.
    Reports OOS R² separately for pre-2008 and post-2008 subperiods to
    distinguish overfitting (bad in-sample fit) from regime break (good
    in-sample but bad out-of-sample post-crisis).
    """
    dep = f"xret_{h}m" if f"xret_{h}m" in panel.columns else f"ret_{h}m"
    df = panel[[vrp_col, dep]].dropna()
    T = len(df)
    y = df[dep].values
    x = df[vrp_col].values

    pred_model = np.full(T, np.nan)
    pred_bench = np.full(T, np.nan)

    first_forecast = window + h - 1
    for t in range(first_forecast, T):
        train_end = t - h + 1  # exclusive; no future return leakage
        train_start = max(0, train_end - window)

        y_train = y[train_start:train_end]
        x_train = x[train_start:train_end]
        pred_bench[t] = float(np.mean(y_train))

        X_train = np.column_stack([np.ones(len(y_train)), x_train])
        try:
            b = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            pred_model[t] = b[0] + b[1] * x[t]
        except Exception:
            pred_model[t] = pred_bench[t]

    def _r2_for_mask(mask: np.ndarray) -> tuple[float, int]:
        m = mask & ~np.isnan(pred_model) & ~np.isnan(pred_bench)
        if m.sum() == 0:
            return np.nan, 0
        e_bench = (y[m] - pred_bench[m]) ** 2
        e_model = (y[m] - pred_model[m]) ** 2
        return float(1.0 - e_model.mean() / e_bench.mean()), int(m.sum())

    valid = ~np.isnan(pred_model) & ~np.isnan(pred_bench)
    dates = df.index
    pre_mask = np.array(dates < pd.Timestamp("2008-01-01"))
    post_mask = np.array(dates >= pd.Timestamp("2008-01-01"))

    oos_full, n_full = _r2_for_mask(valid)
    oos_pre, n_pre = _r2_for_mask(pre_mask & valid)
    oos_post, n_post = _r2_for_mask(post_mask & valid)

    return {
        "OOS_R2_full": oos_full,
        "OOS_R2_pre2008": oos_pre,
        "OOS_R2_post2008": oos_post,
        "N_full": n_full,
        "N_pre": n_pre,
        "N_post": n_post,
        "window": window,
        "predictions": pd.DataFrame({
            "y": y,
            "pred_model": pred_model,
            "pred_bench": pred_bench,
        }, index=df.index),
    }


# ============================================================
# 9. EVT HORSE-RACE
# ============================================================

def evt_horse_race(
    panel: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 6),
    controls: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Regression: xret_{t,t+h} = a + b1 VRP_t + b2 DeltaVRP_t + controls + e.

    DeltaVRP = EVT_VRP - VRP. If b2 is significant, the EVT tail correction
    adds predictive information beyond standard VRP.
    """
    if "EVT_VRP" not in panel.columns:
        print("EVT_VRP not available; run build_evt_rv first.")
        return pd.DataFrame()

    local = panel.copy()
    local["delta_VRP"] = local["EVT_VRP"] - local["VRP"]
    ctrl_cols = list(controls or [])
    results: list[dict[str, object]] = []

    for h in horizons:
        dep = f"xret_{h}m" if f"xret_{h}m" in local.columns else f"ret_{h}m"
        reg_cols = ["VRP", "delta_VRP"] + [c for c in ctrl_cols if c in local.columns]
        df = local[[dep] + reg_cols].dropna()
        if len(df) < 50:
            continue

        y = df[dep].values
        X = sm.add_constant(df[reg_cols].values)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        se = hodrick_se(y, X, h)
        t_stats = safe_tstats(beta_hat, se)

        y_hat = X @ beta_hat
        r2 = 1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

        results.append({
            "horizon": h,
            "beta_VRP": beta_hat[1],
            "t_VRP": t_stats[1],
            "beta_deltaVRP": beta_hat[2],
            "t_deltaVRP": t_stats[2],
            "R2": r2,
            "N": len(df),
            "controls": ", ".join([c for c in ctrl_cols if c in local.columns]) or "none",
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).set_index("horizon")


def evt_horse_race_hac(
    panel: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 6),
    controls: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Same horse-race as evt_horse_race but with Newey-West / HAC standard errors."""
    if "EVT_VRP" not in panel.columns:
        print("EVT_VRP not available; run build_evt_rv first.")
        return pd.DataFrame()

    local = panel.copy()
    local["delta_VRP"] = local["EVT_VRP"] - local["VRP"]
    ctrl_cols = list(controls or [])
    results: list[dict[str, object]] = []

    for h in horizons:
        dep = f"xret_{h}m" if f"xret_{h}m" in local.columns else f"ret_{h}m"
        reg_cols = ["VRP", "delta_VRP"] + [c for c in ctrl_cols if c in local.columns]
        df = local[[dep] + reg_cols].dropna()
        if len(df) < 50:
            continue

        y = df[dep].values
        X = sm.add_constant(df[reg_cols].values)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        se = hac_se(y, X, h)
        t_stats = safe_tstats(beta_hat, se)

        y_hat = X @ beta_hat
        r2 = 1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

        results.append({
            "horizon": h,
            "beta_VRP": beta_hat[1],
            "se_VRP_HAC": se[1],
            "t_VRP_HAC": t_stats[1],
            "beta_deltaVRP": beta_hat[2],
            "se_deltaVRP_HAC": se[2],
            "t_deltaVRP_HAC": t_stats[2],
            "R2": r2,
            "N": len(df),
            "controls": ", ".join([c for c in ctrl_cols if c in local.columns]) or "none",
            "hac_lags": max(int(h) - 1, 0),
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).set_index("horizon")


# ============================================================
# 10. OUTPUT HELPERS AND PLOTS
# ============================================================

def save_table(df: pd.DataFrame, filename: str) -> None:
    if df is None or df.empty:
        return
    ensure_output_dirs()
    df.to_csv(TABLE_DIR / filename)


def plot_vrp_series(panel: pd.DataFrame, filename: str = "fig1_vrp_series.pdf") -> None:
    """Figure 1: VRP, VIX2/RV, and 1-month excess returns with recession shading."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    try:
        fred = make_fred()
        if fred is None:
            raise RuntimeError("fredapi unavailable")
        rec = fred.get_series("USREC", observation_start=panel.index.min(),
                              observation_end=panel.index.max())
        rec = rec.reindex(panel.index, method="ffill").fillna(0)
    except Exception:
        rec = pd.Series(0, index=panel.index)

    def add_recession_shading(ax: plt.Axes, recession: pd.Series) -> None:
        in_rec = False
        start = None
        for date, val in recession.items():
            if val == 1 and not in_rec:
                start = date
                in_rec = True
            elif val == 0 and in_rec and start is not None:
                ax.axvspan(start, date, alpha=0.15, color="gray", lw=0)
                in_rec = False
        if in_rec and start is not None:
            ax.axvspan(start, recession.index[-1], alpha=0.15, color="gray", lw=0)

    axes[0].plot(panel.index, panel["VRP"] * 10000, lw=0.9, label="VRP (standard)")
    if "EVT_VRP" in panel.columns:
        axes[0].plot(panel.index, panel["EVT_VRP"] * 10000, lw=0.9,
                     alpha=0.75, linestyle="--", label="EVT-VRP")
    axes[0].axhline(0, color="black", lw=0.5, linestyle="--")
    add_recession_shading(axes[0], rec)
    axes[0].set_ylabel("VRP (×10⁴)")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Variance Risk Premium — US, BTZ replication + EVT extension")

    axes[1].plot(panel.index, panel["VIX2"] * 10000, lw=0.9, label="VIX²")
    axes[1].plot(panel.index, panel["RV"] * 10000, lw=0.9, label="RV")
    add_recession_shading(axes[1], rec)
    axes[1].set_ylabel("Variance (×10⁴)")
    axes[1].legend(fontsize=9)

    if "xret_1m" in panel.columns:
        axes[2].bar(panel.index, panel["xret_1m"] * 100, width=20, label="1M excess return")
    else:
        axes[2].bar(panel.index, panel["ret_1m"] * 100, width=20, label="1M return")
    add_recession_shading(axes[2], rec)
    axes[2].set_ylabel("Return (%)")
    axes[2].legend(fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    ensure_output_dirs()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_predictability_r2(
    results_univariate: pd.DataFrame,
    results_controls: pd.DataFrame,
    results_evt: Optional[pd.DataFrame] = None,
    filename: str = "fig2_r2_horizons.pdf",
) -> None:
    """Figure 2: In-sample R² across horizons."""
    if results_univariate is None or results_univariate.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results_univariate.index, results_univariate["R2"] * 100,
            "o-", label="VRP only", lw=1.5)

    if results_controls is not None and not results_controls.empty:
        ax.plot(results_controls.index, results_controls["R2"] * 100,
                "s-", label="VRP + controls", lw=1.5)

    if results_evt is not None and not results_evt.empty:
        ax.plot(results_evt.index, results_evt["R2"] * 100,
                "^--", label="VRP + EVT correction", lw=1.5)

    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("In-sample R² (%)")
    ax.set_xticks([1, 3, 6, 12])
    ax.legend()
    ax.set_title("Return predictability of VRP across horizons")
    plt.tight_layout()
    ensure_output_dirs()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_xi(panel: pd.DataFrame, filename: str = "fig3_rolling_xi.pdf") -> None:
    """Figure 3: Rolling GPD tail shape parameter ξ with NBER recession shading.

    ξ = 0 is the exponential boundary (thin tail); ξ = 0.5 is the finite-variance
    boundary — above it the GPD second moment is undefined and the EVT correction
    falls back to observed squared returns.

    Empirically, ξ tends to spike ahead of realized stress events: it rose
    noticeably before the 2008–2009 financial crisis and spiked sharply around
    the March 2020 COVID crash, reflecting the tail thickening that precedes
    large market dislocations.
    """
    if "gpd_xi" not in panel.columns:
        print("gpd_xi not in panel; rebuild the panel or rerun build_us_panel.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    try:
        fred = make_fred()
        if fred is None:
            raise RuntimeError("fredapi unavailable")
        rec = fred.get_series("USREC", observation_start=panel.index.min(),
                              observation_end=panel.index.max())
        rec = rec.reindex(panel.index, method="ffill").fillna(0)
    except Exception:
        rec = pd.Series(0, index=panel.index)

    in_rec = False
    rec_start = None
    for date, val in rec.items():
        if val == 1 and not in_rec:
            rec_start = date
            in_rec = True
        elif val == 0 and in_rec and rec_start is not None:
            ax.axvspan(rec_start, date, alpha=0.15, color="gray", lw=0)
            in_rec = False
    if in_rec and rec_start is not None:
        ax.axvspan(rec_start, rec.index[-1], alpha=0.15, color="gray", lw=0)

    xi_smooth = panel['gpd_xi'].rolling(12, center=True, min_periods=6).mean()
    ax.plot(xi_smooth.index, xi_smooth.values, color='darkorange', linewidth=2.0, alpha=0.9, label='12-month smoothed ξ')
    ax.plot(panel['gpd_xi'].index, panel['gpd_xi'].values, color='steelblue', linewidth=0.8, alpha=0.3, label='GPD ξ (rolling 252-day window)')
    ax.axhline(0.0, color="black", lw=0.9, linestyle="--", label="ξ = 0  (exponential boundary)")
    ax.axhline(0.5, color="crimson", lw=0.9, linestyle="--", label="ξ = 0.5  (finite-variance boundary)")

    ax.set_ylabel("GPD shape  ξ")
    ax.legend(fontsize=9)
    ax.set_title("Rolling GPD tail-shape parameter — S&P 500, 252-day window")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    ensure_output_dirs()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 11. PIPELINE RUNNERS
# ============================================================

def build_us_panel(start: str, end: str, window: int = 252, threshold: float = 0.90) -> pd.DataFrame:
    """Fetch data and build the full US monthly panel."""
    print("Fetching US data...")
    vix_daily = fetch_vix(start, end)
    spx_daily = fetch_spx_returns(start, end)
    macro_raw = fetch_macro_controls(start, end)
    cape_raw = fetch_cape(start, end)

    print(f"VIX: {len(vix_daily)} daily obs ({vix_daily.index.min().date()} – {vix_daily.index.max().date()})")
    print(f"SPX: {len(spx_daily)} daily return obs ({spx_daily.index.min().date()} – {spx_daily.index.max().date()})")

    print("Constructing monthly VIX2 and RV...")
    vix2 = build_monthly_vix2(vix_daily)
    rv_us = build_monthly_rv(spx_daily)

    print("Fitting EVT tail corrections...")
    evt_rv_us, xi_us = build_evt_rv(spx_daily, window=window, threshold_quantile=threshold, return_xi=True)

    print("Building master panel...")
    panel_us = build_panel(vix2, rv_us, evt_rv_us, spx_daily, macro_raw, cape_raw)
    panel_us = panel_us.sort_index()
    panel_us = panel_us.join(xi_us.rename("gpd_xi"), how="left")
    ensure_output_dirs()
    panel_us.to_parquet(DATA_DIR / "panel_us.parquet")
    print(f"Panel: {len(panel_us)} monthly obs ({panel_us.index.min().date()} – {panel_us.index.max().date()})")
    return panel_us


def run_validation_tables(panel: pd.DataFrame, sample: SampleSpec) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Run and save the required validation tables for one sample."""
    p = slice_panel(panel, sample.start, sample.end)
    print("\n" + "=" * 72)
    print(f"SAMPLE: {sample.label} ({sample.start} to {sample.end}) | N={len(p)}")
    print("=" * 72)

    stats = summary_stats(p)
    print("\nTABLE 1: Summary statistics")
    print(stats.to_string() if not stats.empty else "No stats available.")
    save_table(stats, f"table1_summary_{sample.label}.csv")

    res_uni = vrp_predictability(p, vrp_col="VRP", controls=[])
    print("\nTABLE 2A: VRP only")
    print(res_uni.round(4).to_string() if not res_uni.empty else "Insufficient data.")
    save_table(res_uni, f"table2a_vrp_only_{sample.label}.csv")

    res_ctrl = vrp_predictability(
        p,
        vrp_col="VRP",
        controls=["term_spread", "default_spread", "log_cape"],
    )
    print("\nTABLE 2B: VRP + controls")
    print(res_ctrl.round(4).to_string() if not res_ctrl.empty else "Insufficient controls/data.")
    save_table(res_ctrl, f"table2b_vrp_controls_{sample.label}.csv")

    res_evt = evt_horse_race(
        p,
        horizons=[1, 3, 6],
        controls=["term_spread", "default_spread"],
    )
    print("\nTABLE 3: EVT horse-race (Hodrick SE)")
    print(res_evt.round(4).to_string() if not res_evt.empty else "Insufficient EVT/data.")
    save_table(res_evt, f"table3_evt_horserace_{sample.label}.csv")

    res_evt_hac = evt_horse_race_hac(
        p,
        horizons=[1, 3, 6],
        controls=["term_spread", "default_spread"],
    )
    print("\nTABLE 3 HAC/Newey-West robustness")
    print(res_evt_hac.round(4).to_string() if not res_evt_hac.empty else "Insufficient EVT/data.")
    save_table(res_evt_hac, f"table3_evt_horserace_HAC_{sample.label}.csv")

    oos = oos_r2(p, vrp_col="VRP", h=1, min_train=60, return_predictions=True)
    print("\nOOS R², h=1 (expanding window)")
    if np.isfinite(oos["OOS_R2"]):
        print(f"OOS R² = {oos['OOS_R2'] * 100:.2f}% (N={oos['N_eval']})")
    else:
        print("Insufficient data.")
    predictions = oos.get("predictions")
    if isinstance(predictions, pd.DataFrame):
        save_table(predictions, f"oos_predictions_h1_{sample.label}.csv")

    roll_oos = rolling_oos_r2(p, vrp_col="VRP", h=1, window=60)
    print("\nROLLING OOS R², h=1 (60-month window)")

    def _fmt(val: float, n: int) -> str:
        return f"{val * 100:.2f}% (N={n})" if np.isfinite(val) else "Insufficient data."

    print(f"  Full sample: {_fmt(roll_oos['OOS_R2_full'], roll_oos['N_full'])}")
    print(f"  Pre-2008:    {_fmt(roll_oos['OOS_R2_pre2008'], roll_oos['N_pre'])}")
    print(f"  Post-2008:   {_fmt(roll_oos['OOS_R2_post2008'], roll_oos['N_post'])}")

    roll_oos_summary = pd.DataFrame({
        "OOS_R2_pct": [
            roll_oos["OOS_R2_full"] * 100,
            roll_oos["OOS_R2_pre2008"] * 100,
            roll_oos["OOS_R2_post2008"] * 100,
        ],
        "N": [roll_oos["N_full"], roll_oos["N_pre"], roll_oos["N_post"]],
        "window_months": 60,
    }, index=pd.Index(["full", "pre2008", "post2008"], name="period"))
    save_table(roll_oos_summary, f"oos_rolling60_h1_{sample.label}.csv")

    roll_preds = roll_oos.get("predictions")
    if isinstance(roll_preds, pd.DataFrame):
        save_table(roll_preds, f"oos_rolling60_predictions_h1_{sample.label}.csv")

    return {
        "panel": p,
        "stats": stats,
        "res_uni": res_uni,
        "res_ctrl": res_ctrl,
        "res_evt": res_evt,
        "res_evt_hac": res_evt_hac,
        "oos": oos,
        "roll_oos": roll_oos,
    }


def run_australia_extension(start: str, end: str, window: int = 252) -> None:
    """Optional placeholder Australia extension. Not part of the validation-critical path."""
    print("\n" + "=" * 72)
    print("AUSTRALIA EXTENSION")
    print("=" * 72)
    try:
        avix_daily = fetch_avix(start, end)
        asx_daily = fetch_asx_returns(start, end)
    except ImportError as exc:
        print(exc)
        return

    if avix_daily is None:
        print("Skipping Australia: AVIX unavailable from yfinance. Use manual ASX CSV later.")
        return

    avix2 = build_monthly_vix2(avix_daily)
    rv_au = build_monthly_rv(asx_daily)
    evt_rv_au = build_evt_rv(asx_daily, window=window)
    panel_au = build_panel(avix2, rv_au, evt_rv_au, asx_daily, macro=None, cape=None)
    print(f"Australia panel: {len(panel_au)} monthly obs")
    stats_au = summary_stats(panel_au)
    res_au_uni = vrp_predictability(panel_au, vrp_col="VRP", controls=[])
    res_au_evt = evt_horse_race(panel_au, horizons=[1, 3, 6])
    save_table(stats_au, "table1_summary_AU.csv")
    save_table(res_au_uni, "table2a_vrp_only_AU.csv")
    save_table(res_au_evt, "table3_evt_horserace_AU.csv")
    print(stats_au.round(4).to_string())
    print(res_au_uni.round(4).to_string() if not res_au_uni.empty else "Insufficient AU data.")
    print(res_au_evt.round(4).to_string() if not res_au_evt.empty else "Insufficient AU EVT data.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTZ VRP replication validation run")
    parser.add_argument("--start", default=DEFAULT_START, help="Full-sample start date")
    parser.add_argument("--end", default=DEFAULT_END, help="Full-sample end date")
    parser.add_argument("--window", type=int, default=252, help="EVT trailing window in trading days")
    parser.add_argument("--threshold", type=float, default=0.90, help="EVT absolute-return threshold quantile")
    parser.add_argument("--skip-plots", action="store_true", help="Skip PDF figure generation")
    parser.add_argument("--run-australia", action="store_true", help="Attempt optional Australia extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    print("=" * 72)
    print("BTZ VRP Replication | Validation Pass")
    print("=" * 72)
    print(f"Full-sample request: {args.start} to {args.end}")
    print(f"EVT settings: window={args.window}, threshold_quantile={args.threshold}")

    panel_us = build_us_panel(args.start, args.end, window=args.window, threshold=args.threshold)

    samples = [
        SampleSpec("BTZ_1990_2007", BTZ_START, BTZ_END),
        SampleSpec("POST_2008_full", POST_START, args.end),
        SampleSpec("FULL_1990_end", args.start, args.end),
    ]

    outputs = {sample.label: run_validation_tables(panel_us, sample) for sample in samples}

    # ---- Consolidated rolling OOS R² summary (FULL sample, 60-month window) ----
    roll = outputs["FULL_1990_end"]["roll_oos"]
    print("\n" + "=" * 72)
    print("ROLLING 60-MONTH OOS R² SUMMARY (FULL SAMPLE, h=1)")
    print("=" * 72)

    def _pct(v: float) -> str:
        return f"{v * 100:.2f}%" if np.isfinite(v) else "n/a"

    print(f"  Full sample (1990–{args.end[:4]}): {_pct(roll['OOS_R2_full'])}  (N={roll['N_full']})")
    print(f"  1990–2007 subperiod:             {_pct(roll['OOS_R2_pre2008'])}  (N={roll['N_pre']})")
    print(f"  2008–{args.end[:4]} subperiod:            {_pct(roll['OOS_R2_post2008'])}  (N={roll['N_post']})")

    if not args.skip_plots:
        full_panel = outputs["FULL_1990_end"]["panel"]
        plot_vrp_series(full_panel, filename="fig1_vrp_series_FULL_1990_end.pdf")
        plot_predictability_r2(
            outputs["FULL_1990_end"]["res_uni"],
            outputs["FULL_1990_end"]["res_ctrl"],
            outputs["FULL_1990_end"]["res_evt"],
            filename="fig2_r2_horizons_FULL_1990_end.pdf",
        )
        plot_rolling_xi(full_panel, filename="fig3_rolling_xi.pdf")
        print(f"\nFigures saved to {FIGURE_DIR}/")

    if args.run_australia:
        run_australia_extension(args.start, args.end, window=args.window)

    print(f"\nTables saved to {TABLE_DIR}/")
    print(f"Processed panel saved to {DATA_DIR / 'panel_us.parquet'}")
    print("Done.")


if __name__ == "__main__":
    main()
