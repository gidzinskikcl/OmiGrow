#!/bin/bash
set -e

MODALITY="prot"
MODEL_ID="S2"
RESULTS_ROOT="results"
PARAMS_DIR="${RESULTS_ROOT}/predictions/training_${MODEL_ID}_${MODALITY}"
PARAMS_PATH="${PARAMS_DIR}/best_params.json"

# cd to repo root
cd "$(dirname "$0")/../../.."

LOG_DIR="${RESULTS_ROOT}/logs/logs_${MODEL_ID}_${MODALITY}"
LOG_FILE="${LOG_DIR}/training_${MODEL_ID}_${MODALITY}.log"

python scripts/single_view/run_training.py \
    --modality "${MODALITY}" \
    --model_id "${MODEL_ID}" \
    --results_root "${RESULTS_ROOT}/predictions" \
    --params_path "${PARAMS_PATH}" \
    > "${LOG_FILE}" 2>&1

echo "Training completed. Logs saved to: ${LOG_FILE}"
echo "Results are in: ${RESULTS_ROOT}/predictions"