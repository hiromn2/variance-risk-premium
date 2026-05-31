# Variance Risk Premium Replication with EVT Tail Correction

**GitHub:** https://github.com/hiromn2/variance-risk-premium

---

## Motivation

Bollerslev, Tauchen & Zhou (2009) document that the variance risk premium — the difference between option-implied variance (VIX²) and realized variance — is a robust predictor of S&P 500 excess returns at horizons of 3–6 months, with Hodrick-corrected t-statistics above 3.5 in their 1990–2007 sample. Londono (2011) finds that this relationship is substantially weaker in international equity markets, raising the question of whether the predictive power is specific to the US or partly an artifact of how realized variance is measured — particularly in markets where tail events are more frequent and heavier-tailed.

## Method

This project replicates the BTZ predictability regressions on 1990–2026 data and extends the realized variance measure by fitting a Generalized Pareto Distribution to daily return exceedances above a rolling quantile threshold, then replacing each tail day's observed squared return with the GPD conditional second moment:

$$E[r^2 \mid |r| > u] = u^2 + 2u \cdot \frac{\beta}{1-\xi} + \frac{2\beta^2}{(1-\xi)(1-2\xi)}, \quad \xi < \tfrac{1}{2}$$

The horse-race regression tests whether the incremental correction ΔVRPt = EVT\_VRP − VRP predicts returns beyond standard VRP:

$$\text{xret}_{t,t+h} = \alpha + \beta_1 \text{VRP}_t + \beta_2 \Delta\text{VRP}_t + \gamma' \text{controls}_t + \varepsilon_t$$

Inference uses both Hodrick (1992) standard errors — the appropriate estimator for overlapping h-period returns — and Newey-West HAC standard errors (maxlags = h − 1) as a robustness check.

## Results

- **BTZ-period VRP replication (1990–2007):** h=3 Hodrick t-statistic = **3.70**, consistent with the original paper despite the use of daily rather than intraday RV. The result holds across univariate and multivariate specifications with term spread, default spread, and log CAPE controls.

- **EVT correction — BTZ period:** h=1 HAC t-statistic = **1.82** (β = 0.79, correct sign). The EVT tail correction carries a positive signal in the pre-crisis period, consistent with tail risk being priced in the 1990s dot-com and early 2000s recession environment, but falls below conventional significance thresholds.

- **EVT correction — full sample (1990–2026):** HAC t-statistic = −0.54 at h=3, insignificant across all horizons. ΔVRPt has a near-zero mean (−0.015 ×10⁻⁴) relative to a standard deviation of 175 ×10⁻⁴ — the GPD correction adds ~6% to average monthly RV but the increment is contemporaneous noise, not a return-predictive signal in the full sample.

- **Rolling GPD shape parameter ξ:** The 252-day rolling GPD shape (q90 threshold) documents systematic regime variation across 35 years. The Great Moderation (c. 1993–2006) is characterized by a prolonged trough in ξ — near-exponential tail behavior. ξ spikes in the late 1990s (dot-com buildup), the 2008–2009 GFC, and post-2022, confirming that tail risk is time-varying in ways that a fixed distributional assumption would miss.

## Bug Discovery and Why the Null Result Matters

During development, a misconfigured fallback threshold in the GPD rolling estimation — set at < 30 exceedances rather than < 15 — silently triggered on every call: a 252-day window at the q90 quantile produces only ~25 exceedances (252 × 0.10), so the guard always fired and stored ξ = 0 throughout the sample. This produced a spurious full-sample HAC t-statistic of 4.34 for the EVT correction at h=3. The bug was identified by examining the parquet panel, confirming gpd\_xi was identically zero, and tracing the logic back to the threshold comparison. After correcting the guard, the t-statistic collapsed to −0.54. Reporting the corrected null result — rather than the spurious positive finding — is the scientifically correct outcome: the EVT correction is well-specified and theoretically motivated, but the data do not support the hypothesis that GPD tail shape carries incremental return-predictive information in the full post-1990 US sample.

## What This Project Demonstrates

- Implementation of GPD maximum likelihood estimation and the closed-form conditional second moment for generalized Pareto exceedances, including valid-range guards (ξ < 0.5 for finite variance, fallback to exponential when exceedances < 15)
- Reproducible empirical pipeline: rolling estimation with leakage-safe OOS indexing, Hodrick (1992) and Newey-West HAC inference, threshold/window sensitivity grids, and NBER-shaded publication-quality figures
- Understanding of EVT theory sufficient to both implement and diagnose a subtle rolling-window calibration failure
- Intellectual honesty: the pipeline was built to find a positive result; it found a null result; the null result is reported

## References

Bollerslev, Tauchen & Zhou (2009). *Expected stock returns and variance risk premia.* RFS 22(11).  
Campbell & Thompson (2008). *Predicting excess stock returns out of sample.* RFS 21(4).  
Hodrick (1992). *Dividend yields and expected stock returns.* RFS 5(3).  
Londono (2011). *The variance risk premium around the world.* Federal Reserve IFDP 1035.
