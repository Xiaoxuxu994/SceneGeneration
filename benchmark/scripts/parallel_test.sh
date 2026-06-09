#!/usr/bin/env bash
set -euo pipefail

export pred_z_depth=True

MODEL=reliev3r  # ['vggt', 'pi3', 'reliev3r']
MODEL_SIZE=""
ENCODER_TYPE=dinov3  # ['dinov2', 'dinov3']

SRC_ROOT=/efs-m3ga/y00959808/exp/reliev3r
DST_ROOT=experiments

STEPS=(5000 10000)

DST_EXP_NAMES=(
    0608_5Bv2_v9-5_enc1e-1x_dec1x
)

DATASETS=(
    7scenes
    ase
    dl3dv
    dtu
    eth3d
    kitti
    nrgbd
    re10k
    scannet
    scannetpp
)

NPU_IDS=(2 3 5 6 7)
MAX_JOBS_PER_NPU=2
NUM_NPUS=${#NPU_IDS[@]}
NUM_WORKERS=$(( NUM_NPUS * MAX_JOBS_PER_NPU ))

LOG_DIR=logs/eval_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
mkdir -p "$DST_ROOT"

for EXP_NAME in "${DST_EXP_NAMES[@]}"; do
    SRC_DIR="$SRC_ROOT/$EXP_NAME"
    DST_DIR="$DST_ROOT/$EXP_NAME"

    if [[ ! -e "$DST_DIR" ]]; then
        ln -s "$SRC_DIR" "$DST_DIR"
    fi
done

TASKS=()

for EXP_NAME in "${DST_EXP_NAMES[@]}"; do
    for STEP in "${STEPS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            TASKS+=("${EXP_NAME}|${STEP}|${DATASET}")
        done
    done
done

echo "Total tasks: ${#TASKS[@]}"
echo "NPU workers: ${NPU_IDS[*]}"
echo "Log dir: $LOG_DIR"

TASK_COUNTER=0

run_one_task() {
    local EXP_NAME="$1"
    local STEP="$2"
    local DATASET="$3"
    local NPU_ID="$4"
    local TASK_ID="$5"

    local DST_DIR="$DST_ROOT/$EXP_NAME"
    local CKPT="$DST_DIR/ckpt/$STEP"
    local CONFIG="$DST_DIR/config.json"
    local PREFIX="${EXP_NAME}_step${STEP}"
    local LOG_FILE="$LOG_DIR/${EXP_NAME}_step${STEP}_${DATASET}_npu${NPU_ID}.log"

    if [[ ! -e "$CKPT" ]]; then
        echo "[ERROR ${TASK_ID}/${#TASKS[@]}] ckpt not found: $CKPT" >&2
        return 1
    fi

    if [[ ! -f "$CONFIG" ]]; then
        echo "[ERROR ${TASK_ID}/${#TASKS[@]}] config not found: $CONFIG" >&2
        return 1
    fi

    echo "[START ${TASK_ID}/${#TASKS[@]}] npu=${NPU_ID} exp=${EXP_NAME} step=${STEP} dataset=${DATASET}"

    if ! bash "scripts/${ENCODER_TYPE}/run_${DATASET}.sh" \
        "$MODEL" \
        "$MODEL_SIZE" \
        "$CONFIG" \
        "$CKPT" \
        "$PREFIX" \
        "$NPU_ID" \
        > "$LOG_FILE" 2>&1; then

        echo "[FAIL ${TASK_ID}/${#TASKS[@]}] npu=${NPU_ID} exp=${EXP_NAME} step=${STEP} dataset=${DATASET}, log: $LOG_FILE" >&2
        return 1
    fi

    echo "[DONE ${TASK_ID}/${#TASKS[@]}] npu=${NPU_ID} exp=${EXP_NAME} step=${STEP} dataset=${DATASET}, log: $LOG_FILE"
}

run_worker() {
    local WORKER_IDX="$1"
    local NPU_ID="${NPU_IDS[$WORKER_IDX % $NUM_NPUS]}"

    local i
    for ((i = WORKER_IDX; i < ${#TASKS[@]}; i += NUM_WORKERS)); do
        local TASK="${TASKS[$i]}"
        local EXP_NAME STEP DATASET

        IFS='|' read -r EXP_NAME STEP DATASET <<< "$TASK"

        run_one_task "$EXP_NAME" "$STEP" "$DATASET" "$NPU_ID" "$i"
    done
}

PIDS=()

for ((w = 0; w < NUM_WORKERS; w++)); do
    run_worker "$w" &
    PIDS+=("$!")
done

STATUS=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        STATUS=1
    fi
done

if [[ "$STATUS" -ne 0 ]]; then
    echo "[ERROR] Some tasks failed. Check logs in $LOG_DIR" >&2
    exit "$STATUS"
fi

echo "[ALL DONE] logs saved to $LOG_DIR"