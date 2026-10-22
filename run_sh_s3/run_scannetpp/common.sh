#!/usr/bin/env bash
# Shared ModelArts launcher for the ScanNet++-only loop-transformer ablation.
# Do not run directly: use run_exp1022_0*_*.sh, which set NAME and MODEL.
# Everything except NAME / MODEL (data, training recipe, loss weights, resources) is identical for all runs.
set -euo pipefail

: "${NAME:?set NAME in the wrapper script}"
: "${MODEL:?set MODEL (configs/model/<MODEL>.yaml) in the wrapper script}"

# -----------------------------
# Paths (edit here if your buckets / EFS differ)
# -----------------------------
# Repo root = two levels above this script, wherever the job's code directory is (user-job-dir/Pi3,
# user-job-dir/SceneGeneration, a dev checkout, ...). Override with CODE_DIR=... if needed.
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CKPT_LINK_SRC=/efs-mi-east4-2/checkpoints                     # -> ./ckpts (DINOv3 weights live here)
DINOV3_S_WEIGHTS=ckpts/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
OBS_OUT_ROOT="obs://bucket-mi-east4-sf/z00533370/project/proj_3dgs/exp_pi3sup/pi3sup/outputs"
EFS_OUT_ROOT="/efs-m3ga/z00533370/exp/pi3sup/outputs"
TRAIN_URL="/home/ma-user/modelarts/outputs/train_url_0"

# Loss weights (same as the latest from-scratch runs, e.g. exp0603); identical for A and B.
W_UNPROJ_GLOBAL=1.0
W_NORMALIZED=1.0

# Smoke test: SMOKE=1 bash run_X.sh -> 1 epoch x 50 iters, 5 test iters, separate output name.
if [ "${SMOKE:-0}" = "1" ]; then
    NAME="${NAME}_smoke"
    EXTRA_ARGS=(train.num_epoch=1 train.iters_per_epoch=50 test.iters_per_test=5 log.ckpt_interval=1)
else
    EXTRA_ARGS=()
fi

echo "==> [${NAME}] model=${MODEL} data=scannetpp_only train=train_scannetpp_dinov3 code=${CODE_DIR}"

cd "${CODE_DIR}"
if [ ! -f scripts/train_pi3.py ]; then
    echo "[ERROR] ${CODE_DIR} is not the SceneGeneration repo root (scripts/train_pi3.py not found)" >&2
    exit 1
fi
export PYTHONPATH="${CODE_DIR}"
if [ ! -e ckpts ]; then
    ln -s "${CKPT_LINK_SRC}" ckpts
fi
if [ ! -f "${DINOV3_S_WEIGHTS}" ]; then
    # torch.hub would otherwise try to download from the internet
    echo "[ERROR] DINOv3 ViT-S/16 weights not found: $(pwd)/${DINOV3_S_WEIGHTS}" >&2
    exit 1
fi
mkdir -p "${TRAIN_URL}"

for pkg in hydra_colorlog evo pyquaternion; do
    python -c "import ${pkg}" 2>/dev/null || python -m pip install "${pkg}"
done

copy_by_mox () {
    local SRC="$1" DST="$2"
    if [[ "${DST}" != obs://* ]]; then mkdir -p "${DST}"; fi
    echo "==> mox copy: ${SRC} -> ${DST}"
    python - <<PY
import moxing as mox
mox.file.copy_parallel(r"""${SRC}""", r"""${DST}""")
print("[mox] copy done")
PY
}

# Ascend / HCCL
export DEVICE_TYPE=ascend
export TOKENIZERS_PARALLELISM=false
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export USE_MEMARTS=0
export USE_CHECKPOINT=True          # gradient checkpointing inside the DINOv3 encoder
export HYDRA_FULL_ERROR=1

# Multi-node (16 NPUs = 2 nodes x 8)
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="${MA_NUM_HOSTS}"
NODE_RANK="${VC_TASK_INDEX}"
NGPUS_PER_NODE="${MA_NUM_GPUS}"
NUM_PROCESSES=$((NGPUS_PER_NODE * NNODES))
echo "MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NGPUS_PER_NODE=${NGPUS_PER_NODE} NUM_PROCESSES=${NUM_PROCESSES}"
if [ "${NUM_PROCESSES}" != "16" ]; then
    echo "[WARN] recipe is sized for 16 NPUs (got ${NUM_PROCESSES}); images/step and LR are not rescaled"
fi

# -----------------------------
# Auto-resume: copy the latest checkpoint from OBS
# -----------------------------
OBS_CKPT_ROOT="${OBS_OUT_ROOT}/${NAME}/outputs/${NAME}/ckpts"
LOCAL_CKPT_ROOT="${TRAIN_URL}/outputs/${NAME}/ckpts"
LATEST_CKPT_FILE="/tmp/latest_ckpt_${NAME}.txt"
mkdir -p "${LOCAL_CKPT_ROOT}"
rm -f "${LATEST_CKPT_FILE}"
python - <<PY
import re
import moxing as mox
root, out = r"""${OBS_CKPT_ROOT}""", r"""${LATEST_CKPT_FILE}"""
latest, latest_id = None, -1
if mox.file.exists(root):
    for item in mox.file.list_directory(root):
        path = item if item.startswith("obs://") else root.rstrip("/") + "/" + item.rstrip("/")
        m = re.match(r"checkpoint_(\\d+)$", path.rstrip("/").split("/")[-1])
        if m and int(m.group(1)) > latest_id:
            latest_id, latest = int(m.group(1)), path
if latest:
    open(out, "w").write(latest)
print(f"[AUTO_RESUME] latest ckpt: {latest}")
PY
if [ -s "${LATEST_CKPT_FILE}" ]; then
    REMOTE_CKPT="$(cat "${LATEST_CKPT_FILE}")"
    copy_by_mox "${REMOTE_CKPT}" "${LOCAL_CKPT_ROOT}/$(basename "${REMOTE_CKPT}")"
fi

# -----------------------------
# Train
# -----------------------------
echo "==> Start training"
accelerate launch \
  --config_file configs/accelerate/ddp.yaml \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port 29540 \
  --machine_rank "${NODE_RANK}" \
  --num_machines "${NNODES}" \
  --num_processes "${NUM_PROCESSES}" \
  scripts/train_pi3.py \
  model="${MODEL}" \
  train=train_scannetpp_dinov3 \
  data=scannetpp_only \
  name="${NAME}" \
  work_dir="${TRAIN_URL}" \
  loss.train_loss.w_unproj_global=${W_UNPROJ_GLOBAL} \
  loss.train_loss.w_normalized_loss=${W_NORMALIZED} \
  loss.test_loss.w_unproj_global=${W_UNPROJ_GLOBAL} \
  loss.test_loss.w_normalized_loss=${W_NORMALIZED} \
  log.ckpt_interval=2 \
  log.max_checkpoints=99999 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
echo "==> Training finished"

# -----------------------------
# Upload: latest checkpoint (every node) + tensorboard (global rank 0)
# -----------------------------
GLOBAL_RANK="${RANK:-${RANK_ID:-${OMPI_COMM_WORLD_RANK:-0}}}"
CKPT_DIR=$(ls -d "${TRAIN_URL}/outputs/${NAME}/ckpts/checkpoint_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -n "${CKPT_DIR}" ]; then
    copy_by_mox "${CKPT_DIR}" "${EFS_OUT_ROOT}/${NAME}/ckpts/$(basename "${CKPT_DIR}")"
fi
LOG_DIR="${TRAIN_URL}/outputs/${NAME}/${NAME}"
if [ "${GLOBAL_RANK}" = "0" ] && [ -d "${LOG_DIR}" ]; then
    copy_by_mox "${LOG_DIR}" "${EFS_OUT_ROOT}/${NAME}/${NAME}"
fi
