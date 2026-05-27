"""
Variance Risk Premium Replication & EVT Extension
Based on: Bollerslev, Tauchen & Zhou (2009), RFS
Extension: EVT-corrected realized variance, motivated by Londono (2011) puzzle

Structure:
  0. Dependencies
  1. Data acquisition  (US + Australia)
  2. Realized variance construction  (standard + EVT-corrected)
  3. VRP construction
  4. Summary statistics  (Table 1 of BTZ)
  5. Predictability regressions  (Table 2 of BTZ) with Hodrick (1992) SEs
  6. Out-of-sample R²  (Campbell-Thompson 2008)
  7. EVT extension  (GPD tail correction + horse-race regressions)
  8. Plots

Usage:
  pip install yfinance pandas_datareader fredapi scipy statsmodels matplotlib
  python btz_vrp_replication.py

  Or paste cells into a Jupyter notebook — each section is self-contained.
"""

# ============================================================
# 0. DEPENDENCIES
# ============================================================
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from scipy.stats import genpareto
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file instead of displaying
import matplotlib.pyplot as plt
import os
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

matplotlib.use("Agg")

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})

START = "1990-01-01"
END   = "2007-12-31"


# ============================================================
# 1. DATA ACQUISITION
# ============================================================

def fetch_vix(start=START, end=END):
    """
    CBOE VIX daily close. Free from Yahoo Finance (^VIX).
    VIX is in annualised vol units (percent). We need variance: (VIX/100)^2 * 12
    to express as monthly variance in decimal^2 terms — matching BTZ Table 1 units.
    
    Note: CBOE also provides raw VIX history CSV at
    https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
    Use that as a backup if yfinance has gaps.
    """
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"].squeeze()
    vix.name = "VIX"
    return vix


def fetch_spx_returns(start=START, end=END):
    """
    S&P 500 daily log-returns from Yahoo Finance (^GSPC).
    We use log-returns throughout: r_t = log(P_t / P_{t-1}).
    """
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"].squeeze()
    ret = np.log(spx / spx.shift(1)).dropna()
    ret.name = "spx_logret"
    return ret


def fetch_asx_returns(start=START, end=END):
    """
    ASX 200 daily log-returns (^AXJO).
    Note: yfinance coverage for ^AXJO starts ~2000.
    """
    asx = yf.download("^AXJO", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"].squeeze()
    ret = np.log(asx / asx.shift(1)).dropna()
    ret.name = "asx_logret"
    return ret


def fetch_avix(start=START, end=END):
    """
    ASX 200 VIX (^AVIX on some platforms). yfinance coverage is limited.
    Fallback: S&P/ASX 200 VIX data from ASX website (manual CSV download).
    If ^AVIX fails, we approximate implied variance from 1m ATM options
    or use the CBOE VXASX index if available on your data provider.

    For now we attempt ^AVIX; if empty, returns None and the extension
    section will note the limitation.
    """
    try:
        avix = yf.download("^AVIX", start=start, end=end, auto_adjust=True,
                           progress=False)["Close"].squeeze()
        if avix.empty or len(avix.dropna()) < 100:
            print("WARNING: ^AVIX data sparse. Australia VRP requires manual "
                  "download from ASX website or Bloomberg.")
            return None
        avix.name = "AVIX"
        return avix
    except Exception as e:
        print(f"AVIX fetch failed: {e}")
        return None


def fetch_macro_controls(start=START, end=END):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))  # no API key needed for basic access
    series = {
        "gs10": "GS10",
        "tb3m": "TB3MS",
        "baa":  "BAA",
        "aaa":  "AAA",
    }
    frames = {}
    for name, fred_id in series.items():
        try:
            s = fred.get_series(fred_id, observation_start=start,
                                observation_end=end)
            s.name = name
            frames[name] = s
        except Exception as e:
            print(f"FRED fetch failed for {fred_id}: {e}")

    macro = pd.DataFrame(frames)
    macro["term_spread"]    = macro["gs10"] - macro["tb3m"]
    macro["default_spread"] = macro["baa"]  - macro["aaa"]
    return macro


def fetch_cape(start=START, end=END):
    """
    Shiller CAPE (P/E10) — free from Robert Shiller's website.
    URL: http://www.econ.yale.edu/~shiller/data/ie_data.xls
    We use log(P/E) following BTZ.

    Parsing note: the Excel file has an irregular header; row 7 is the
    actual column header in the current version.
    """
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    try:
        df = pd.read_excel(url, sheet_name="Data", header=7)
        # Column names in Shiller file
        df = df[["Date", "P", "E"]].copy()
        df = df[df["Date"].notna() & df["P"].notna()].copy()
        # Date column is decimal year (e.g. 1990.01 = Jan 1990)
        df["Date"] = pd.to_datetime(
            df["Date"].astype(str).str[:7],
            format="%Y.%m", errors="coerce"
        )
        df = df.dropna(subset=["Date"]).set_index("Date")
        # 10-year trailing average earnings
        df["E10"] = df["E"].rolling(120).mean()
        df["P"]  = pd.to_numeric(df["P"],  errors="coerce")
        df["E10"] = pd.to_numeric(df["E10"], errors="coerce")
        df["cape"] = np.log(df["P"] / df["E10"])
        cape = df["cape"].dropna()
        cape.index = cape.index.to_period("M").to_timestamp("M")
        return cape
    except Exception as e:
        print(f"CAPE fetch failed: {e}. Skipping CAPE control.")
        return None


# ============================================================
# 2. REALIZED VARIANCE CONSTRUCTION
# ============================================================

def build_monthly_rv(daily_returns: pd.Series) -> pd.Series:
    """
    Standard realized variance: sum of squared daily log-returns within
    each calendar month, annualised by ×12.

    BTZ use intraday (5-min) data for higher precision. With daily data
    we lose some accuracy but the predictability result is qualitatively
    robust (confirmed by later papers using daily data).

    Units: annualised variance, decimal^2 (so ~0.04 for a 20% vol market)
    """
    monthly = (
        daily_returns
        .groupby(daily_returns.index.to_period("M"))
        .apply(lambda x: (x**2).sum() * 12)     # annualise
    )
    monthly.index = monthly.index.to_timestamp("M")
    monthly.name = "RV"
    return monthly


def build_monthly_vix2(daily_vix: pd.Series) -> pd.Series:
    """
    VRP uses month-end VIX observation as the implied variance for the
    coming month. VIX is in annualised vol % units.

    Convert: VIX_t^2 / 10000 gives annualised variance in decimal^2.
    Take month-end value (last trading day of month).
    """
    # Month-end value
    monthly = (
        daily_vix
        .groupby(daily_vix.index.to_period("M"))
        .last()
    )
    monthly.index = monthly.index.to_timestamp("M")
    vix2 = (monthly / 100) ** 2          # annualised variance, decimal^2
    vix2.name = "VIX2"
    return vix2


# ============================================================
# 3. EVT-CORRECTED REALIZED VARIANCE  (your extension)
# ============================================================

def fit_gpd_tail(returns: np.ndarray, threshold_quantile: float = 0.90):
    """
    Fit a Generalized Pareto Distribution to the exceedances of |r| above
    the threshold_quantile quantile.

    The GPD has CDF:
        F(x) = 1 - (1 + ξ x/β)^{-1/ξ}   for ξ ≠ 0

    Parameters
    ----------
    returns : np.ndarray of daily log-returns
    threshold_quantile : float, threshold as quantile of |r|

    Returns
    -------
    xi : float, shape parameter (> 0 means heavy right tail)
    beta : float, scale parameter
    u : float, threshold value
    """
    abs_r = np.abs(returns)
    u = np.quantile(abs_r, threshold_quantile)
    exceedances = abs_r[abs_r > u] - u

    if len(exceedances) < 30:
        # Too few exceedances; return Gaussian fallback
        return 0.0, np.std(abs_r), u

    # MLE via scipy
    xi, loc, beta = genpareto.fit(exceedances, floc=0)
    return xi, beta, u


def evt_tail_variance(xi: float, beta: float, u: float,
                      n_obs: int, n_exceed: int) -> float:
    """
    Analytical expected squared return contribution from the tail |r| > u,
    under a GPD(ξ, β) model for the exceedances.

    E[r² · 1(|r|>u)] = P(|r|>u) · E[r² | |r|>u]

    For a GPD exceedance y = |r| - u ~ GPD(ξ, β):
        E[(y+u)²] = E[y²] + 2u·E[y] + u²

    GPD moments (for ξ < 1/2 for variance to exist):
        E[y]  = β / (1 - ξ)
        E[y²] = 2β² / ((1-ξ)(1-2ξ))
    """
    p_exceed = n_exceed / n_obs  # empirical exceedance probability (both tails: ×2)
    p_exceed *= 2                # symmetric: account for negative returns too

    if xi >= 0.5:
        # Variance doesn't exist under GPD with ξ >= 0.5 — use empirical fallback
        return np.nan

    ey  = beta / (1 - xi)
    ey2 = 2 * beta**2 / ((1 - xi) * (1 - 2*xi))
    e_r2_given_exceed = ey2 + 2 * u * ey + u**2

    return p_exceed * e_r2_given_exceed


def build_evt_rv(daily_returns: pd.Series,
                 window: int = 252,
                 threshold_quantile: float = 0.90) -> pd.Series:
    """
    Build EVT-corrected monthly realized variance.

    For each month t:
      1. Estimate GPD on the trailing `window` daily returns.
      2. Compute the body contribution to RV: sum of r² for |r| ≤ u.
      3. Replace the tail contribution (sum of r² for |r| > u) with the
         analytical GPD expectation.
      4. Annualise ×12.

    This reduces the noise from large squared returns dominating RV
    in fat-tailed markets (the Londono puzzle mechanism).
    """
    rets = daily_returns.values
    dates = daily_returns.index
    monthly_periods = daily_returns.groupby(daily_returns.index.to_period("M"))

    evt_rv_list = []
    evt_rv_dates = []

    for period, group in monthly_periods:
        month_end_idx = dates.get_loc(group.index[-1])
        if month_end_idx < window:
            continue

        # Trailing window for GPD estimation
        train = rets[month_end_idx - window: month_end_idx]
        xi, beta, u = fit_gpd_tail(train, threshold_quantile)

        # Current month returns
        r_month = group.values
        abs_r = np.abs(r_month)

        # Body contribution: |r| ≤ u, computed directly
        body_rv = np.sum(r_month[abs_r <= u] ** 2)

        # Tail contribution: analytical GPD expectation
        n_obs    = len(train)
        n_exceed = np.sum(np.abs(train) > u)
        tail_days = np.sum(abs_r > u)        # observed tail days this month

        # Scale analytical expectation to number of tail days observed
        tail_rv_analytical = evt_tail_variance(xi, beta, u, n_obs, n_exceed)

        if np.isnan(tail_rv_analytical):
            # Fallback to empirical for this month
            tail_rv = np.sum(r_month[abs_r > u] ** 2)
        else:
            # Weight: analytical expectation × number of tail days
            # (We replace each tail r² with its expectation under GPD)
            tail_rv = tail_days * (u**2 + tail_rv_analytical / max(n_exceed/n_obs*2, 1e-6))

        evt_rv_list.append((body_rv + tail_rv) * 12)   # annualise
        evt_rv_dates.append(period.to_timestamp("M"))

    evt_rv = pd.Series(evt_rv_list, index=pd.DatetimeIndex(evt_rv_dates),
                       name="EVT_RV")
    return evt_rv


# ============================================================
# 4. BUILD MASTER PANEL
# ============================================================

def build_panel(vix2: pd.Series, rv: pd.Series, evt_rv: pd.Series,
                spx_ret: pd.Series, macro: pd.DataFrame,
                cape: pd.Series = None) -> pd.DataFrame:
    """
    Merge all monthly series into a single panel.

    Forward returns at horizon h computed here for h=1,3,6,12.
    VRP is lagged 1 month relative to returns (predetermined predictor).
    """
    # Monthly log-returns (sum daily log-rets within month, then annualise ×12)
    monthly_ret = (
        spx_ret
        .groupby(spx_ret.index.to_period("M"))
        .sum()
    )
    monthly_ret.index = monthly_ret.index.to_timestamp("M")
    monthly_ret.name = "ret_1m"

    panel = pd.concat([vix2, rv, evt_rv, monthly_ret], axis=1).dropna(
        subset=["VIX2", "RV"])

    # VRP (standard) and EVT-VRP
    panel["VRP"]     = panel["VIX2"] - panel["RV"]
    panel["EVT_VRP"] = panel["VIX2"] - panel["EVT_RV"]

    # Merge macro controls (monthly FRED data)
    if macro is not None:
        macro_m = macro.resample("ME").last()
        panel = panel.join(macro_m[["term_spread", "default_spread"]],
                           how="left")

    # Merge CAPE
    if cape is not None:
        panel = panel.join(cape.rename("log_cape"), how="left")
        panel["log_cape"] = panel["log_cape"].ffill()

    # Forward returns at multiple horizons (non-overlapping for h=1,
    # overlapping for h>1 — standard in the literature)
    for h in [1, 3, 6, 12]:
        panel[f"ret_{h}m"] = (
            monthly_ret
            .rolling(h)
            .sum()
            .shift(-h)     # future h-month return
        )

    # Excess returns (approximate: subtract 1m T-bill rate)
    # We use term_spread as a proxy for rf here; ideally use TB3MS directly
    # Subtract annualised rf / 12 per month
    if "tb3m" in macro.columns:
        rf_monthly = macro["tb3m"].resample("ME").last() / 100 / 12
        panel = panel.join(rf_monthly.rename("rf"), how="left")
        for h in [1, 3, 6, 12]:
            panel[f"xret_{h}m"] = panel[f"ret_{h}m"] - panel["rf"] * h

    return panel.dropna(subset=["VRP"])


# ============================================================
# 5. SUMMARY STATISTICS  (replicates BTZ Table 1)
# ============================================================

def summary_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Report mean, std, skewness, excess kurtosis, AR(1) for
    VRP, RV, VIX2. Compare against BTZ Table 1.
    
    BTZ Table 1 benchmarks (1990-2007):
      VRP:  mean ~0.00236, std ~0.00249
      RV:   mean ~0.0155,  std ~0.0170
      VIX2: mean ~0.0179,  std ~0.0178
    """
    cols = ["VRP", "RV", "VIX2", "EVT_RV", "EVT_VRP"]
    cols = [c for c in cols if c in panel.columns]
    stats = {}
    for c in cols:
        s = panel[c].dropna()
        stats[c] = {
            "mean":     s.mean(),
            "std":      s.std(),
            "skew":     s.skew(),
            "kurt":     s.kurtosis(),     # excess kurtosis
            "AR(1)":    s.autocorr(lag=1),
            "min":      s.min(),
            "max":      s.max(),
            "N":        len(s),
        }
    return pd.DataFrame(stats).T.round(5)


# ============================================================
# 6. HODRICK (1992) STANDARD ERRORS
# ============================================================

def hodrick_se(y: np.ndarray, X: np.ndarray, h: int) -> np.ndarray:
    """
    Hodrick (1992) standard errors for overlapping return regressions.

    When the dependent variable is the h-period ahead return, OLS
    residuals are serially correlated (MA(h-1) structure). Newey-West
    over-corrects in small samples. Hodrick's method is the standard in
    this literature.

    Algorithm (Hodrick 1992, equation (10)):
      1. Regress y on X via OLS to get residuals e.
      2. Construct S = (1/T) * X'(h·I + Γ_1 + ... + Γ_{h-1})X where
         Γ_j = (1/T) Σ_{t=j+1}^T e_t * e_{t-j} * X_t * X_{t-j}'
      3. V(β̂) = (X'X/T)^{-1} S (X'X/T)^{-1}

    Parameters
    ----------
    y : (T,) array of dependent variable (h-period forward return)
    X : (T, k) array of regressors (including constant)
    h : forecast horizon (months)

    Returns
    -------
    se : (k,) array of standard errors
    """
    T, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta

    # (X'X / T)
    XtX_inv = np.linalg.inv(X.T @ X / T)

    # Γ_0
    S = np.zeros((k, k))
    for t in range(T):
        S += np.outer(e[t] * X[t], e[t] * X[t])
    S = S / T

    # Γ_j for j = 1,...,h-1
    for j in range(1, h):
        G = np.zeros((k, k))
        for t in range(j, T):
            G += np.outer(e[t] * X[t], e[t-j] * X[t-j])
        G = G / T
        S += G + G.T

    V = XtX_inv @ S @ XtX_inv / T
    se = np.sqrt(np.diag(V))
    return se


# ============================================================
# 7. PREDICTABILITY REGRESSIONS  (replicates BTZ Table 2)
# ============================================================

def vrp_predictability(panel: pd.DataFrame,
                       horizons: list = [1, 3, 6, 12],
                       vrp_col: str = "VRP",
                       controls: list = None) -> pd.DataFrame:
    """
    Run OLS regression:
        xret_{t,t+h} = α + β·VRP_t + γ·controls_t + ε

    Report OLS β, Hodrick SE, t-stat, R² (in-sample).
    Uses complete cases per regression.

    BTZ Table 2 benchmarks (to check your replication):
      h=1:  β~2.50 (t~2.3), R²~3%
      h=3:  β~3.80 (t~3.1), R²~7%
      h=6:  β~2.90 (t~2.4), R²~5%
      h=12: β~1.20 (t~1.0), R²~2%
    """
    results = []
    ctrl_cols = controls or []

    for h in horizons:
        dep = f"xret_{h}m"
        if dep not in panel.columns:
            dep = f"ret_{h}m"   # fallback to raw returns

        reg_cols = [vrp_col] + ctrl_cols
        reg_cols = [c for c in reg_cols if c in panel.columns]

        df = panel[[dep] + reg_cols].dropna()
        if len(df) < 50:
            continue

        y = df[dep].values
        X = sm.add_constant(df[reg_cols].values)

        # OLS
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

        # Hodrick SEs
        se = hodrick_se(y, X, h)
        t_stats = beta_hat / se

        # R²
        y_hat = X @ beta_hat
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot

        results.append({
            "horizon": h,
            "beta_VRP":   beta_hat[1],
            "se_VRP":     se[1],
            "t_VRP":      t_stats[1],
            "R2":         r2,
            "N":          len(df),
            "controls":   ", ".join(ctrl_cols) if ctrl_cols else "none",
        })

    return pd.DataFrame(results).set_index("horizon")


# ============================================================
# 8. OUT-OF-SAMPLE R²  (Campbell & Thompson 2008)
# ============================================================

def oos_r2(panel: pd.DataFrame,
           vrp_col: str = "VRP",
           h: int = 1,
           min_train: int = 60) -> dict:
    """
    Campbell-Thompson (2008) out-of-sample R²:

        OOS-R² = 1 - MSE_model / MSE_benchmark

    where benchmark is the expanding-window historical mean return.
    Uses recursive (expanding window) estimation — no future information.

    Returns dict with OOS-R² and a time series of cumulative ΔSSE
    (for the Clark-West test plot).
    """
    dep = f"xret_{h}m" if f"xret_{h}m" in panel.columns else f"ret_{h}m"
    df = panel[[vrp_col, dep]].dropna()

    T = len(df)
    y = df[dep].values
    X_vrp = df[vrp_col].values

    pred_model = np.full(T, np.nan)
    pred_bench = np.full(T, np.nan)

    for t in range(min_train, T - h):
        # Benchmark: expanding mean
        pred_bench[t] = y[:t].mean()

        # Model: OLS on expanding window
        yt = y[:t]
        Xt = np.column_stack([np.ones(t), X_vrp[:t]])
        try:
            b = np.linalg.lstsq(Xt, yt, rcond=None)[0]
            pred_model[t] = b[0] + b[1] * X_vrp[t]
        except Exception:
            pred_model[t] = pred_bench[t]

    # Evaluate on forecast period
    valid = ~np.isnan(pred_model)
    e_bench = (y[valid] - pred_bench[valid])**2
    e_model = (y[valid] - pred_model[valid])**2

    oos_r2_val = 1 - e_model.mean() / e_bench.mean()

    cum_dmse = np.cumsum(e_bench - e_model)

    return {
        "OOS_R2": oos_r2_val,
        "cum_dmse": pd.Series(cum_dmse,
                              index=df.index[valid]),
        "N_eval": valid.sum(),
    }


# ============================================================
# 9. EVT HORSE-RACE  (your original extension)
# ============================================================

def evt_horse_race(panel: pd.DataFrame,
                   horizons: list = [1, 3, 6],
                   controls: list = None) -> pd.DataFrame:
    """
    Test whether EVT-corrected VRP has incremental predictive power
    beyond standard VRP.

    Regression:
        xret_{t,t+h} = α + β₁·VRP_t + β₂·ΔVRP_t + controls + ε

    where ΔVRP_t = EVT_VRP_t - VRP_t is the "tail correction" component.
    If β₂ > 0 and significant, the EVT adjustment adds information.

    This directly addresses Londono's puzzle: does correcting for
    tail-induced RV noise improve predictability?
    """
    if "EVT_VRP" not in panel.columns:
        print("EVT_VRP not available; run build_evt_rv first.")
        return pd.DataFrame()

    panel = panel.copy()
    panel["delta_VRP"] = panel["EVT_VRP"] - panel["VRP"]

    results = []
    ctrl_cols = controls or []

    for h in horizons:
        dep = f"xret_{h}m" if f"xret_{h}m" in panel.columns else f"ret_{h}m"
        reg_cols = ["VRP", "delta_VRP"] + [c for c in ctrl_cols
                                            if c in panel.columns]
        df = panel[[dep] + reg_cols].dropna()
        if len(df) < 50:
            continue

        y = df[dep].values
        X = sm.add_constant(df[reg_cols].values)

        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        se       = hodrick_se(y, X, h)
        t_stats  = beta_hat / se

        y_hat = X @ beta_hat
        r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)

        results.append({
            "horizon":         h,
            "beta_VRP":        beta_hat[1],
            "t_VRP":           t_stats[1],
            "beta_deltaVRP":   beta_hat[2],
            "t_deltaVRP":      t_stats[2],
            "R2":              r2,
            "N":               len(df),
        })

    return pd.DataFrame(results).set_index("horizon")


# ============================================================
# 10. PLOTS
# ============================================================

def plot_vrp_series(panel: pd.DataFrame, save: bool = False):
    """
    Figure 1: VRP over time with NBER recession shading.
    Replicates BTZ Figure 1 conceptually.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Fetch NBER recession dates from FRED
    try:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        rec = fred.get_series("USREC", observation_start=panel.index.min(),
                       observation_end=panel.index.max())
        rec = rec.reindex(panel.index, method="ffill").fillna(0)
    except Exception:
        rec = pd.Series(0, index=panel.index)

    def add_recession_shading(ax, rec, index):
        in_rec = False
        for i, (date, val) in enumerate(rec.items()):
            if val == 1 and not in_rec:
                start = date
                in_rec = True
            elif val == 0 and in_rec:
                ax.axvspan(start, date, alpha=0.15, color="gray", lw=0)
                in_rec = False

    # Panel A: VRP
    axes[0].plot(panel.index, panel["VRP"] * 10000, color="#534AB7",
                 lw=0.9, label="VRP (std)")
    if "EVT_VRP" in panel.columns:
        axes[0].plot(panel.index, panel["EVT_VRP"] * 10000, color="#EF9F27",
                     lw=0.9, alpha=0.7, linestyle="--", label="EVT-VRP")
    axes[0].axhline(0, color="black", lw=0.5, linestyle="--")
    add_recession_shading(axes[0], rec, panel.index)
    axes[0].set_ylabel("VRP (×10⁴)")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Variance Risk Premium — US (BTZ replication + EVT extension)")

    # Panel B: VIX² vs RV
    axes[1].plot(panel.index, panel["VIX2"] * 10000, color="#0F6E56",
                 lw=0.9, label="VIX² (implied var)")
    axes[1].plot(panel.index, panel["RV"] * 10000, color="#D85A30",
                 lw=0.9, label="RV (realised var)")
    add_recession_shading(axes[1], rec, panel.index)
    axes[1].set_ylabel("Variance (×10⁴)")
    axes[1].legend(fontsize=9)

    # Panel C: 1M S&P excess return
    if "xret_1m" in panel.columns:
        axes[2].bar(panel.index, panel["xret_1m"] * 100, color="#888780",
                    width=20, label="Excess return (1M)")
        axes[2].set_ylabel("Return (%)")
    add_recession_shading(axes[2], rec, panel.index)
    axes[2].legend(fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    if save:
        plt.savefig("fig1_vrp_series.pdf", bbox_inches="tight")
    plt.show()


def plot_predictability_r2(results_univariate: pd.DataFrame,
                            results_controls: pd.DataFrame,
                            results_evt: pd.DataFrame = None,
                            save: bool = False):
    """
    Figure 2: In-sample R² across horizons for three specifications.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    horizons = results_univariate.index

    ax.plot(horizons, results_univariate["R2"] * 100, "o-",
            color="#534AB7", label="VRP only", lw=1.5)
    ax.plot(horizons, results_controls["R2"] * 100, "s-",
            color="#0F6E56", label="VRP + controls", lw=1.5)
    if results_evt is not None and len(results_evt):
        ax.plot(results_evt.index, results_evt["R2"] * 100, "^--",
                color="#EF9F27", label="VRP + EVT correction", lw=1.5)

    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("In-sample R² (%)")
    ax.set_xticks([1, 3, 6, 12])
    ax.legend()
    ax.set_title("Return predictability of VRP across horizons")
    plt.tight_layout()
    if save:
        plt.savefig("fig2_r2_horizons.pdf", bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN: run the full pipeline
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BTZ VRP Replication  |  Fetching data...")
    print("=" * 60)

    # --- Fetch ---
    vix_daily  = fetch_vix()
    spx_daily  = fetch_spx_returns()
    asx_daily  = fetch_asx_returns()
    avix_daily = fetch_avix()
    macro_raw  = fetch_macro_controls()
    cape_raw   = fetch_cape()

    print(f"VIX:  {len(vix_daily)} daily obs  "
          f"({vix_daily.index.min().date()} – {vix_daily.index.max().date()})")
    print(f"SPX:  {len(spx_daily)} daily obs")
    print(f"ASX:  {len(asx_daily)} daily obs")

    # --- Construct monthly series ---
    print("\nConstructing monthly realized variance...")
    vix2    = build_monthly_vix2(vix_daily)
    rv_us   = build_monthly_rv(spx_daily)

    print("Fitting EVT tail corrections (this takes ~30s on a full sample)...")
    evt_rv_us = build_evt_rv(spx_daily, window=252, threshold_quantile=0.90)

    # --- Build panel ---
    print("Building master panel...")
    panel_us = build_panel(vix2, rv_us, evt_rv_us, spx_daily,
                           macro_raw, cape_raw)
    print(f"Panel: {len(panel_us)} monthly obs, "
          f"{panel_us.index.min().date()} – {panel_us.index.max().date()}")

    # --- Summary statistics ---
    print("\n" + "="*60)
    print("TABLE 1: Summary Statistics")
    print("="*60)
    stats = summary_stats(panel_us)
    print(stats.to_string())
    print("\nBTZ benchmarks (1990-2007):")
    print("  VRP mean ~0.00236, std ~0.00249")
    print("  RV  mean ~0.0155,  std ~0.0170")

    # --- Core regressions ---
    print("\n" + "="*60)
    print("TABLE 2A: Univariate VRP predictability (Hodrick SEs)")
    print("="*60)
    res_uni = vrp_predictability(panel_us, vrp_col="VRP",
                                  controls=[])
    print(res_uni.round(4).to_string())

    print("\nTABLE 2B: VRP + controls")
    res_ctrl = vrp_predictability(
        panel_us, vrp_col="VRP",
        controls=["term_spread", "default_spread", "log_cape"]
    )
    print(res_ctrl.round(4).to_string())

    # --- OOS R² ---
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE R² (Campbell-Thompson, h=1)")
    print("="*60)
    oos = oos_r2(panel_us, vrp_col="VRP", h=1, min_train=60)
    print(f"OOS R² = {oos['OOS_R2']*100:.2f}%  (N={oos['N_eval']})")

    # --- EVT extension ---
    print("\n" + "="*60)
    print("TABLE 3: EVT horse-race (ΔVRP incremental predictability)")
    print("="*60)
    res_evt = evt_horse_race(
        panel_us,
        controls=["term_spread", "default_spread"]
    )
    print(res_evt.round(4).to_string())

    # --- Plots ---
    print("\nGenerating figures...")
    plot_vrp_series(panel_us, save=True)
    plot_predictability_r2(res_uni, res_ctrl, res_evt, save=True)

    # --- Australia (if AVIX available) ---
    if avix_daily is not None:
        print("\n" + "="*60)
        print("AUSTRALIA VRP (extension)")
        print("="*60)
        avix2   = build_monthly_vix2(avix_daily)
        rv_au   = build_monthly_rv(asx_daily)
        evt_rv_au = build_evt_rv(asx_daily, window=252)

        # Placeholder macro for Australia (use US spreads as proxy,
        # or source from RBA: https://www.rba.gov.au/statistics/)
        panel_au = build_panel(avix2, rv_au, evt_rv_au, asx_daily,
                               macro_raw, cape_raw)
        print(f"Australia panel: {len(panel_au)} obs")

        stats_au = summary_stats(panel_au)
        print(stats_au[["mean","std","skew","kurt"]].round(5))

        res_au_uni = vrp_predictability(panel_au, vrp_col="VRP",
                                         controls=[])
        print("\nAustralia VRP predictability (Londono null expected):")
        print(res_au_uni.round(4))

        res_au_evt = evt_horse_race(panel_au, horizons=[1, 3, 6])
        print("\nAustralia EVT horse-race:")
        print(res_au_evt.round(4))

    print("\nDone. Figures saved to working directory.")
