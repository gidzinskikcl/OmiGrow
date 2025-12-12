#!/bin/bash
set -e

MODALITY_1="expr"         # expr / prot
MODALITY_2="prot"         # expr / prot
MODEL_ID="MF1"         
RESULTS_ROOT="results"

WEIGHTS_PATH_1="weights/S1_expr_7.weights.h5"
WEIGHTS_PATH_2="weights/S2_prot_19.weights.h5"
PARAMS_PATH_1="results/predictions/training_S1_expr/best_params.json"
PARAMS_PATH_2="results/predictions/training_S2_prot/best_params.json"

PARAMS_DIR="${RESULTS_ROOT}/predictions/training_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}"
PARAMS_PATH="${PARAMS_DIR}/best_params.json"
# cd to repo root
cd "$(dirname "$0")/../../../.."

# Log directory
LOG_DIR="${RESULTS_ROOT}/logs/logs_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}"
LOG_FILE="${LOG_DIR}/training_${MODEL_ID}_${MODALITY_1}_${MODALITY_2}.log"

python scripts/multi_view/middle_fusion/run_training.py \
    --modality_1 "${MODALITY_1}" \
    --modality_2 "${MODALITY_2}" \
    --model_id "${MODEL_ID}" \
    --weights_1_path "${WEIGHTS_PATH_1}" \
    --weights_2_path "${WEIGHTS_PATH_2}" \
    --modality_1_params_path  "${PARAMS_PATH_1}" \
    --modality_2_params_path  "${PARAMS_PATH_2}" \
    --results_root "${RESULTS_ROOT}/predictions" \
    --params_path "${PARAMS_PATH}" \
    > "${LOG_FILE}" 2>&1

echo "Training completed. Logs saved to: ${LOG_FILE}"
echo "Results are in: ${RESULTS_ROOT}/predictions"