# Bollerslev, Tauchen & Zhou (2009) — Variance Risk Premium Replication with EVT Extension

## Abstract

Bollerslev, Tauchen & Zhou (2009) show that the variance risk premium (VRP = VIX² − RV) predicts S&P 500 excess returns with Hodrick-corrected t-statistics above 3.5 at horizons of 3–6 months, a finding that has motivated a large literature on the risk premium embedded in option-implied variance. This project replicates the BTZ result on 1990–2026 data and extends it by replacing raw squared daily returns on tail days with the GPD conditional second moment E[r² | |r| > u], motivated by Londono's (2011) observation that the VRP–return relationship is weaker internationally — a pattern potentially explained by differences in tail shape. The extended horse-race regression tests whether ΔVRPt = EVT\_VRP − VRP carries incremental predictive power beyond standard VRP. The BTZ-period (1990–2007) replication is clean: h=3 Hodrick t-statistic of 3.70, consistent with the original paper. The EVT correction shows a positive but statistically insignificant signal in the BTZ period (h=1 HAC t = 1.82) and is entirely insignificant in the full 1990–2026 sample. A critical bug discovered during development — a GPD fallback threshold set too high (< 30 exceedances) that silently triggered on every rolling window, storing ξ = 0 throughout — initially produced a spurious h=3 HAC t-statistic of 4.34 for the EVT correction. After fixing the threshold to < 15 (the correct floor for a 252-day/q90 rolling window yielding ~25 exceedances), the signal collapsed to −0.54. The corrected null result is reported here.

## Key Results

| Result | Sample | Horizon | Statistic |
|--------|--------|---------|-----------|
| VRP return predictability (BTZ replication) | 1990–2007 | h = 3 | Hodrick t = **3.70** |
| VRP return predictability (full sample) | 1990–2026 | h = 3 | Hodrick t = 0.43 |
| EVT tail correction — incremental predictability | 1990–2007 | h = 1 | HAC t = **1.82** (correct sign, below significance) |
| EVT tail correction — incremental predictability | 1990–2026 | h = 3 | HAC t = −0.54 (insignificant) |

The structural break in VRP predictability post-2008 — a well-documented phenomenon — is confirmed: rolling 60-month OOS R² deteriorates from −3.2% (1990–2007) to −58.9% (2008–2026).

## Rolling GPD Shape Parameter ξ

The rolling GPD shape parameter (252-day window, q90 threshold) shows clear regime variation across the 35-year sample. The Great Moderation (roughly 1993–2006) is associated with a prolonged trough in ξ, reflecting bounded, near-exponential tail behavior during a period of low return volatility. Spikes in ξ — indicating heavier tails and higher tail risk — are concentrated in the late 1990s (dot-com buildup), the 2008–2009 financial crisis, and a renewed elevation post-2022, consistent with a return to heavier-tailed equity return distributions.

## Data Sources

| Series | Source | Free |
|--------|--------|------|
| VIX daily | CBOE via Yahoo Finance (`^VIX`) | ✓ |
| S&P 500 returns | Yahoo Finance (`^GSPC`) | ✓ |
| Term / default spread | FRED (`GS10`, `TB3MS`, `BAA`, `AAA`) | ✓ |
| Shiller CAPE | [Shiller website](http://www.econ.yale.edu/~shiller/data/) | ✓ |
| NBER recession dates | FRED (`USREC`) | ✓ |

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export FRED_API_KEY=your_key    # free at fred.stlouisfed.org
python scripts/btz_vrp_replication.py
```

Outputs are written to `reports/tables/` (CSV) and `reports/figures/` (PDF). The processed panel is cached at `data/processed/panel_us.parquet`; delete it to force a fresh download.

Optional arguments:

```bash
python scripts/btz_vrp_replication.py --end 2024-12-31   # custom end date
python scripts/btz_vrp_replication.py --threshold 0.95   # EVT quantile
python scripts/btz_vrp_replication.py --window 504       # rolling window (days)
python scripts/btz_vrp_replication.py --skip-plots       # skip figure generation
```

Threshold–window sensitivity grids (q85/q90/q95 × w252/w504/w756) can be run with:

```bash
bash scripts/run_threshold_sensitivity.sh
```

Unit tests (offline, no API keys required):

```bash
python -m pytest tests/
```

## Limitations

- **Daily RV proxy.** Realized variance is constructed from daily squared returns rather than intraday 5-minute returns. Daily-frequency RV is a noisier proxy and likely understates true integrated variance, which may attenuate VRP estimates and weaken predictability results relative to the original BTZ paper.
- **CAPE disabled in horse-race.** Shiller CAPE is included in the summary statistics and univariate controls but is excluded from the EVT horse-race regression to avoid losing observations at the start of the sample where CAPE and EVT\_RV overlap is limited.
- **EVT correction is insignificant in the full sample.** ΔVRPt = EVT\_VRP − VRP has a near-zero mean (−0.015 ×10⁻⁴) relative to a standard deviation of 175 ×10⁻⁴. The GPD correction adds roughly 6% to average monthly RV but the increment is contemporaneous noise uncorrelated with future returns in the 1990–2026 sample.
- **Negative out-of-sample R².** VRP produces negative OOS R² in both the full sample and post-2008 subperiod under both expanding and 60-month rolling windows, indicating that the historical mean is a better out-of-sample forecast than the VRP-based model in the post-crisis period.

## References

Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies*, 22(11), 4463–4492.

Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? *Review of Financial Studies*, 21(4), 1509–1531.

Hodrick, R. J. (1992). Dividend yields and expected stock returns: Alternative procedures for inference and measurement. *Review of Financial Studies*, 5(3), 357–386.

Londono, J. M. (2011). The variance risk premium around the world. *Federal Reserve International Finance Discussion Paper 1035*.
