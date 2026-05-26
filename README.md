# Bollerslev, Tauchen & Zhou (2009) — Variance Risk Premium Replication

Replication of:

> Bollerslev, T., Tauchen, G., & Zhou, H. (2009).
> *Expected Stock Returns and Variance Risk Premia.*
> Review of Financial Studies, 22(11), 4463–4492.

Extended with:
- EVT-corrected realized variance (GPD tail adjustment)
- Australian market application (addressing Londono 2011 puzzle)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Structure
## Data sources

| Series | Source | Free |
|--------|--------|------|
| VIX daily | CBOE via Yahoo Finance (`^VIX`) | ✓ |
| S&P 500 returns | Yahoo Finance (`^GSPC`) | ✓ |
| ASX 200 returns | Yahoo Finance (`^AXJO`) | ✓ |
| Term / default spread | FRED (`GS10`, `TB3MS`, `BAA`, `AAA`) | ✓ |
| Shiller CAPE | [Shiller website](http://www.econ.yale.edu/~shiller/data/) | ✓ |

## Replication status

- [ ] Data pipeline
- [ ] Realized variance construction
- [ ] VRP series (Table 1)
- [ ] Predictability regressions (Table 2)
- [ ] Hodrick (1992) standard errors
- [ ] Out-of-sample R²
- [ ] EVT extension
- [ ] Australia comparison

## Reference

Londono, J.M. (2011). *The Variance Risk Premium Around the World.*
Federal Reserve IFDP 1035. [Motivates the EVT extension.]
