#!/bin/bash
set -e

MODALITY="prot"
MODEL_ID="S2"
RESULTS_ROOT="results"

cd "$(dirname "$0")/.."

LOG_DIR="${RESULTS_ROOT}/logs/logs_${MODEL_ID}_${MODALITY}"
LOG_FILE="${LOG_DIR}/training_${MODEL_ID}_${MODALITY}.log"

python scripts/run_single_view_training.py \
    --modality "${MODALITY}" \
    --model_id "${MODEL_ID}" \
    --results_root "${RESULTS_ROOT}" \
    > "${LOG_FILE}" 2>&1

echo "Training completed. Logs saved to: ${LOG_FILE}"
