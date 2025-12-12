#!/bin/bash
set -e

MODALITY_1="expr"         # expr / prot
MODALITY_2="prot"         # expr / prot
MODEL_ID="EF1"         
RESULTS_ROOT="results"
PARAMS_DIR="${RESULTS_ROOT}/predictions/training_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}"
PARAMS_PATH="${PARAMS_DIR}/best_params.json"

# cd to repo root
cd "$(dirname "$0")/../../../.."

# Log directory
LOG_DIR="${RESULTS_ROOT}/logs/logs_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}"
LOG_FILE="${LOG_DIR}/training_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}.log"

python scripts/multi_view/early_fusion/run_training.py \
    --model_id "${MODEL_ID}" \
    --modality_1 "${MODALITY_1}" \
    --modality_2 "${MODALITY_2}" \
    --results_root "${RESULTS_ROOT}/predictions" \
    --params_path "${PARAMS_PATH}" \
    > "${LOG_FILE}" 2>&1

echo "Training completed. Logs saved to: ${LOG_FILE}"