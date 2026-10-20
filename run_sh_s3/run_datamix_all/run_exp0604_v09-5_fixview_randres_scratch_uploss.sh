#!/usr/bin/env bash
set -euo pipefail

echo "==> Start multi-node Pi3 training"

cd /home/ma-user/modelarts/user-job-dir/Pi3
export PYTHONPATH=/home/ma-user/modelarts/user-job-dir/Pi3

# 可选：软链接 checkpoint 目录
if [ ! -e ckpts ]; then
    ln -s /efs-mi-east4-2/checkpoints ckpts
fi

# output
TRAIN_URL="/home/ma-user/modelarts/outputs/train_url_0"
mkdir -p "$TRAIN_URL"
echo "TRAIN_URL=$TRAIN_URL"

# 可选：安装 hydra_colorlog
python - <<'PY'
try:
    import hydra_colorlog
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hydra_colorlog"])
PY

python - <<'PY'
try:
    import evo
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "evo"])
PY

python - <<'PY'
try:
    import pyquaternion
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyquaternion"])
PY

# # -----------------------------
# mox function
# -----------------------------
copy_by_mox () {
    local SRC="$1"
    local DST="$2"

    if [[ "${DST}" != obs://* ]]; then
        mkdir -p "${DST}"
    fi
    echo "==> mox copy: ${SRC} -> ${DST}"

    python - <<PY
import moxing as mox
src = r"""${SRC}"""
dst = r"""${DST}"""
mox.file.copy_parallel(src, dst)
print(f"[mox] copy done: {src} -> {dst}")
PY
}
# # -----------------------------
# # mox dataset:
# # -----------------------------
# DATA_wildrgbd_SRC="obs://bucket-beijing-graphics/y00855328/sr_datasets/processed/wildrgbd/"
# DATA_wildrgbd_DST="/cache/data/wildrgbd"

# echo "==> Start moxing ${DATA_wildrgbd_SRC} -> ${DATA_wildrgbd_DST}"
# copy_by_mox "${DATA_wildrgbd_SRC}" "${DATA_wildrgbd_DST}"
# echo "==> Finish moxing wildrgbd"
# ls -lah "${DATA_wildrgbd_DST}" | head || true



# 关键环境
export DEVICE_TYPE=ascend
export TOKENIZERS_PARALLELISM=false
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export USE_MEMARTS=0

# 多机参数
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="${MA_NUM_HOSTS}"
NODE_RANK="${VC_TASK_INDEX}"
NGPUS_PER_NODE="${MA_NUM_GPUS}"
NUM_PROCESSES=$((NGPUS_PER_NODE * NNODES))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"

export USE_CHECKPOINT=False
# export ASCEND_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
NAME="exp0604_pi3_3B_datamix_all_v9-5_randresNew_scratch_uploss_pinmemTrue"

# 实验结果（checkpoint、tensorboard）上传到贵阳 efs：${EXP_ROOT}/pi3sup/outputs/${NAME}/
EXP_ROOT="/efs-mi-pay-gy1/z00533370/exp"
mkdir -p "${EXP_ROOT}/pi3sup/outputs/${NAME}"

# -----------------------------
# Auto copy latest ckpt from OBS to local output dir
# -----------------------------
AUTO_RESUME=1
OBS_CKPT_ROOT="obs://bucket-mi-east4-sf/z00533370/project/proj_3dgs/exp_pi3sup/pi3sup/outputs/${NAME}/outputs/${NAME}/ckpts"
LOCAL_CKPT_ROOT="${TRAIN_URL}/outputs/${NAME}/ckpts"
LATEST_CKPT_FILE="/tmp/latest_ckpt_${NAME}.txt"

mkdir -p "${LOCAL_CKPT_ROOT}"
rm -f "${LATEST_CKPT_FILE}"

if [ "${AUTO_RESUME}" = "1" ]; then
    echo "==> Checking latest OBS checkpoint: ${OBS_CKPT_ROOT}"

    python - <<PY
import re
import moxing as mox

root = r"""${OBS_CKPT_ROOT}"""
out = r"""${LATEST_CKPT_FILE}"""

latest = None
latest_id = -1

if mox.file.exists(root):
    for item in mox.file.list_directory(root):
        path = item if item.startswith("obs://") else root.rstrip("/") + "/" + item.rstrip("/")
        name = path.rstrip("/").split("/")[-1]
        m = re.match(r"checkpoint_(\\d+)$", name)
        if m and int(m.group(1)) > latest_id:
            latest_id = int(m.group(1))
            latest = path

if latest:
    with open(out, "w") as f:
        f.write(latest)
    print(f"[AUTO_RESUME] latest ckpt: {latest}")
else:
    print("[AUTO_RESUME] no checkpoint found")
PY

    if [ -s "${LATEST_CKPT_FILE}" ]; then
        REMOTE_CKPT="$(cat "${LATEST_CKPT_FILE}")"
        LOCAL_CKPT="${LOCAL_CKPT_ROOT}/$(basename "${REMOTE_CKPT}")"

        echo "==> Copy latest ckpt:"
        echo "    ${REMOTE_CKPT}"
        echo " -> ${LOCAL_CKPT}"

        copy_by_mox "${REMOTE_CKPT}" "${LOCAL_CKPT}"
        ls -lah "${LOCAL_CKPT_ROOT}" || true
    fi
fi

# -----------------------------
# Training Process
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
  train=train_pi3_highres \
  data=datamix_all_v9-5 \
  name="${NAME}" \
  work_dir="${TRAIN_URL}" \
  model.use_global_points=False \
  model.freeze_encoder=false \
  model.num_dec_blk_not_to_checkpoint=16 \
  train.optimizer.lr=1e-4 \
  train.optimizer.encoder_lr=1e-5 \
  train.lr_scheduler.pct_start=0.05 \
  train.lr_scheduler.div_factor=25 \
  train.lr_scheduler.final_div_factor=100 \
  train.num_workers=4 \
  test.num_workers=4 \
  train.num_epoch=80 \
  train.iters_per_epoch=2500 \
  test.iters_per_test=20 \
  train.print_freq=20 \
  log.ckpt_interval=1 \
  log.max_checkpoints=99999 \
  'train.image_num_range=[8,8]' \
  train.max_img_per_gpu=8 \
  train.random_reslution=True \
  'train.aspect_ratio_range=[0.5, 2.0]' \
  'train.pixel_count_range=[100000, 255000]' \
  train.patch_size=16 \
  train.num_resolution=-1 \
  'train.width_range=[400,600]' \
  loss.train_loss.w_unproj_global=1.0 \
  loss.train_loss.w_normalized_loss=1.0 \
  loss.test_loss.w_unproj_global=1.0 \
  loss.test_loss.w_normalized_loss=1.0

echo "==> Training finished"

# model.ckpt=/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt79 \

# 获取 global rank，兼容 torchrun / ModelArts / MPI
GLOBAL_RANK="${RANK:-${RANK_ID:-${OMPI_COMM_WORLD_RANK:-0}}}"

CKPT_DIR=$(ls -d ${TRAIN_URL}/outputs/${NAME}/ckpts/checkpoint_* | sort -V | tail -1)
CKPT_EFS_DIR="${EXP_ROOT}/pi3sup/outputs/${NAME}/ckpts/$(basename ${CKPT_DIR})"

echo "==> Start uploading checkpoint"
copy_by_mox "${CKPT_DIR}" "${CKPT_EFS_DIR}"
echo "==> Finish uploading checkpoint"

LOG_EFS_DIR="${EXP_ROOT}/pi3sup/outputs/${NAME}/${NAME}"
LOG_DIR="${TRAIN_URL}/outputs/${NAME}/${NAME}"

# tensorboard 只在 global rank 0 上传
if [ "${GLOBAL_RANK}" = "0" ]; then
    echo "==> Start uploading tensorboard log"
    if [ -d "${LOG_DIR}" ]; then
        copy_by_mox "${LOG_DIR}" "${LOG_EFS_DIR}"
        echo "==> Finish uploading tensorboard log"
    else
        echo "[WARN] Tensorboard log dir not found, skip: ${LOG_DIR}"
    fi
else
    echo "[rank ${GLOBAL_RANK}] skip uploading tensorboard log"
fi
