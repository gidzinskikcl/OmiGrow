#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODALITY_1="expr"         # expr / prot
MODALITY_2="prot"         # expr / prot
MODEL_ID="EF1"           # logical model ID (prefix)
RESULTS_ROOT="results"  # where to store results
N_TRIALS=50
N_SPLITS=5
MAX_EPOCHS=300 

# cd to repo root
cd "$(dirname "$0")/../../../.."

# Log directory
LOG_DIR="${RESULTS_ROOT}/logs/logs_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/tuning_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}.log"

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

python scripts/multi_view/early_fusion/run_hyperparameter_optimisation.py \
    --model_id "${MODEL_ID}" \
    --modality_1 "${MODALITY_1}" \
    --modality_2 "${MODALITY_2}" \
    --results_root "${RESULTS_ROOT}/predictions" \
    --n_trials "${N_TRIALS}" \
    --n_splits "${N_SPLITS}" \
    --max_epochs "${MAX_EPOCHS}" \
    > "${LOG_FILE}" 2>&1

echo
echo "Completed for ${MODEL_ID} (${MODALITY_1}+${MODALITY_2})."
echo "Results are in: ${RESULTS_ROOT}/predictions"
echo "Logs are in: ${LOG_DIR}"