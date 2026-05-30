#!/usr/bin/env bash
set -euo pipefail

END_DATE="${1:-2026-05-29}"
WINDOW="${2:-756}"

mkdir -p reports/tables/threshold_sensitivity
mkdir -p reports/figures/threshold_sensitivity

for THRESHOLD in 0.85 0.90 0.95
do
    TAG="q$(python - <<PY
thr = float("$THRESHOLD")
print(int(round(thr * 100)))
PY
)_w${WINDOW}"

    echo "============================================================"
    echo "Running threshold sensitivity: threshold=${THRESHOLD}, window=${WINDOW}, tag=${TAG}"
    echo "============================================================"

    python scripts/btz_vrp_replication.py \
        --end "${END_DATE}" \
        --threshold "${THRESHOLD}" \
        --window "${WINDOW}"

    for SAMPLE in BTZ_1990_2007 POST_2008_full FULL_1990_end
    do
        cp "reports/tables/table1_summary_${SAMPLE}.csv" \
           "reports/tables/threshold_sensitivity/table1_summary_${SAMPLE}_${TAG}.csv"

        cp "reports/tables/table3_evt_horserace_${SAMPLE}.csv" \
           "reports/tables/threshold_sensitivity/table3_evt_horserace_${SAMPLE}_${TAG}.csv"

        cp "reports/tables/table3_evt_horserace_HAC_${SAMPLE}.csv" \
           "reports/tables/threshold_sensitivity/table3_evt_horserace_HAC_${SAMPLE}_${TAG}.csv"
    done

    cp "reports/figures/fig1_vrp_series_FULL_1990_end.pdf" \
       "reports/figures/threshold_sensitivity/fig1_vrp_series_FULL_1990_end_${TAG}.pdf"

    cp "reports/figures/fig2_r2_horizons_FULL_1990_end.pdf" \
       "reports/figures/threshold_sensitivity/fig2_r2_horizons_FULL_1990_end_${TAG}.pdf"

done

echo "Threshold sensitivity runs complete."
