set -e

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODALITY="expr"         # expr / prot
MODEL_ID="S1"           # logical model ID (prefix)
RESULTS_ROOT="results"  # where to store results
N_TRIALS=50
N_SPLITS=5
MAX_EPOCHS=300 

# cd to repo root (directory containing this script, then up one level)
cd "$(dirname "$0")/../../.."

# Log directory
LOG_DIR="${RESULTS_ROOT}/logs/logs_${MODEL_ID}_${MODALITY}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

LOG_FILE="${LOG_DIR}/tuning_${MODEL_ID}_${MODALITY}.log"

python scripts/single_view/run_hyperparameter_optimisation.py \
    --modality "${MODALITY}" \
    --model_id "${MODEL_ID}" \
    --results_root "${RESULTS_ROOT}/predictions" \
    --n_trials "${N_TRIALS}" \
    --n_splits "${N_SPLITS}" \
    --max_epochs "${MAX_EPOCHS}" \
    > "${LOG_FILE}" 2>&1

echo
echo "Completed for ${MODEL_ID} (${MODALITY})."
echo "Results are in: ${RESULTS_ROOT}/predictions"
echo "Log is in: ${LOG_FILE}"