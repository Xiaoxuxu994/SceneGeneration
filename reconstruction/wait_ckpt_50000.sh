#!/usr/bin/env bash
set -euo pipefail

CKPT_ROOT="/efs-m3ga/c50041401/exp/ssff3d/Reliev3R-260709-1B-Pi3-stage2/ckpt"
CKPT_ID="50000"
CKPT_DIR="${CKPT_ROOT}/${CKPT_ID}"

# zero_pp_rank_0 ~ zero_pp_rank_63
NUM_ZERO_RANKS=64

# 如果你的文件名是 zero_pp_rank_0_mp_rank_00_optim_states.pt，
# 把这里改成 "00"
MP_RANK_SUFFIX="00"

CHECK_INTERVAL=600          # 每 600 秒检查一次
EXTRA_WAIT_AFTER_READY=600  # 满足条件后额外等 10 分钟

echo "[WAIT] Waiting for checkpoint directory: ${CKPT_DIR}"

check_ckpt_ready() {
    if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "[WAIT] Missing checkpoint dir: ${CKPT_DIR}"
        return 1
    fi

    local model_file="${CKPT_DIR}/mp_rank_00_model_states.pt"
    if [[ ! -f "${model_file}" ]]; then
        echo "[WAIT] Missing model state: ${model_file}"
        return 1
    fi

    for i in $(seq 0 $((NUM_ZERO_RANKS - 1))); do
        local optim_file="${CKPT_DIR}/zero_pp_rank_${i}_mp_rank_${MP_RANK_SUFFIX}_optim_states.pt"
        if [[ ! -f "${optim_file}" ]]; then
            echo "[WAIT] Missing optimizer state: ${optim_file}"
            return 1
        fi
    done

    return 0
}

while true; do
    if check_ckpt_ready; then
        echo "[READY] Checkpoint ${CKPT_ID} is complete."
        echo "[READY] Extra waiting ${EXTRA_WAIT_AFTER_READY}s to avoid partially written files..."
        sleep "${EXTRA_WAIT_AFTER_READY}"
        echo "[READY] Done. Continue to torchrun."
        exit 0
    fi

    echo "[WAIT] Checkpoint not ready. Sleep ${CHECK_INTERVAL}s..."
    sleep "${CHECK_INTERVAL}"
done