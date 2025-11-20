set -e

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODALITY="expr"         # expr / prot / flux
MODEL_ID="S1"           # logical model ID (prefix)
RESULTS_ROOT="results"  # where to store results
CHUNK_SIZE=40           # how many configs per run (slice)

# cd to repo root (directory containing this script, then up one level)
cd "$(dirname "$0")/.."

# Log directory
LOG_DIR="${RESULTS_ROOT}/logs/logs_${MODEL_ID}_${MODALITY}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------------
# Determine total grid size from Python
# ------------------------------------------------------------------
TOTAL_CONFIGS=$(python - << 'EOF'
import sys, os
# ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.grid import grid
print(len(grid))
EOF
)

echo "Total configs in grid: ${TOTAL_CONFIGS}"
echo "Chunk size: ${CHUNK_SIZE}"

# ------------------------------------------------------------------
# Run all slices sequentially
# ------------------------------------------------------------------
START=0

while [ "${START}" -lt "${TOTAL_CONFIGS}" ]; do
    END=$((START + CHUNK_SIZE))
    if [ "${END}" -gt "${TOTAL_CONFIGS}" ]; then
        END=${TOTAL_CONFIGS}
    fi

    echo
    echo "============================================================"
    echo "Running slice: [${START}:${END})"
    echo "============================================================"
    echo

    LOG_FILE="${LOG_DIR}/tuning_${MODEL_ID}_${MODALITY}_${START}_${END}.log"

    python scripts/run_single_view_tuning.py \
        --modality "${MODALITY}" \
        --model_id "${MODEL_ID}" \
        --results_root "${RESULTS_ROOT}" \
        --start "${START}" \
        --end "${END}" \
        > "${LOG_FILE}" 2>&1

    echo "Slice [${START}:${END}) finished. Log saved to: ${LOG_FILE}"

    START=${END}
done

echo
echo "All slices completed for ${MODEL_ID} (${MODALITY})."
echo "Logs are in: ${LOG_DIR}"
