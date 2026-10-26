#!/usr/bin/env bash
# Pi3 3B（dinov3_huge+_3Bv2）在 datamix_all_v9-5-sub00（贵阳 OBS 上现有的 23 个数据集，打开数据过滤开关）上从头训练。
# 训练超参与 run_exp0604_v09-5_fixview_randres_scratch_uploss.sh 完全一致，只换了数据配置和实验名，并适配贵阳环境：
#   - 数据：data=datamix_all_v9-5-sub00（过滤开关见该配置文件头；数据读 OBS；启动前把需要的索引缓存拷到节点本地盘）
#   - ++train.worker_reseed=true：各卡 / 各 worker 用不同随机数流（旧版所有 worker 同一条流，增强和选帧会重复）
#   - 断点续训：同原脚本，从 ModelArts 自动同步到 OBS 的训练输出里找最新 checkpoint（改成贵阳桶，见 OBS_TRAIN_ROOT），
#     另外也查 efs 上传目录，取两边编号最大的
#   - 单机 8 卡时不依赖 ModelArts 的多机环境变量（没有就按单机处理）
#
# 用法：
#   bash run_sh_s3/run_datamix_all/run_exp1023_pi3_3B_datamix_all_v9-5-sub00_filter.sh
# 可用环境变量：
#   CODE_DIR    代码目录（默认本脚本所在仓库）
#   CKPTS_SRC   预训练权重目录，软链接成 ./ckpts（需含 dinov3_vith16plus/）；默认 /efs-mi-east4-2/checkpoints
#   EXP_ROOT    实验结果根目录（默认 /efs-mi-pay-gy1/z00533370/exp）
#   TRAIN_URL   训练时的本地输出目录（默认 /home/ma-user/modelarts/outputs/train_url_0）
#   LOCAL_CACHE 索引缓存的本地目录（默认 /cache/dataset_cache；设成空则直接读 OBS）
#   OBS_TRAIN_ROOT ModelArts 作业「训练输出」train_url_0 对应的 OBS 目录（必须与作业配置一致），
#               默认 obs://bucket-471/z00533370/exp/pi3sup/outputs/<NAME>；checkpoint 在 <OBS_TRAIN_ROOT>/outputs/<NAME>/ckpts/
#   AUTO_RESUME 1 = 从 OBS / efs 上本实验最新的 checkpoint 续训（默认 1）
#   NAME        实验名（默认 exp1023_pi3_3B_datamix_all_v9-5-sub00_filter）
#   UPLOAD      0 = 训练结束后不往 efs 传 checkpoint / tensorboard（冒烟测试用）
# 脚本参数会原样追加到训练命令末尾（hydra 覆盖，后写的生效），例如冒烟测试：
#   NAME=smoke_exp1023 AUTO_RESUME=0 UPLOAD=0 bash <本脚本> train.num_epoch=1 train.iters_per_epoch=20 test.iters_per_test=2 train.print_freq=1
set -euo pipefail

echo "==> Start Pi3 training (datamix_all_v9-5-sub00)"

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}"
echo "CODE_DIR=${CODE_DIR}  ($(git log --oneline -1 2>/dev/null || echo 'not a git repo'))"

# 预训练权重（dinov3_vith16plus）
CKPTS_SRC="${CKPTS_SRC:-/efs-mi-east4-2/checkpoints}"
if [ ! -e ckpts ]; then
    ln -s "${CKPTS_SRC}" ckpts
fi
if ! ls ckpts/dinov3_vith16plus/*.pth >/dev/null 2>&1; then
    echo "[ERROR] 找不到 ckpts/dinov3_vith16plus/*.pth（ckpts -> $(readlink -f ckpts || echo '?')）。"
    echo "        设置 CKPTS_SRC=<含 dinov3_vith16plus/ 的目录> 后重试（已有错误的 ckpts 软链接请先 rm ckpts）。"
    exit 1
fi

# output
TRAIN_URL="${TRAIN_URL:-/home/ma-user/modelarts/outputs/train_url_0}"
mkdir -p "$TRAIN_URL"
echo "TRAIN_URL=$TRAIN_URL"

# 依赖：缺的包才安装（hydra-core、hydra_colorlog、evo、pyquaternion 等，见 run_sh_s3/ensure_deps.sh）
bash run_sh_s3/ensure_deps.sh

# -----------------------------
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

# -----------------------------
# 索引缓存拷到本地盘（每个节点一次；同名同大小的跳过）
# -----------------------------
DATA_CFG="configs/data/datamix_all_v9-5-sub00.yaml"
LOCAL_CACHE="${LOCAL_CACHE-/cache/dataset_cache}"
if [ -n "${LOCAL_CACHE}" ]; then
    echo "==> Prefetch dataset caches -> ${LOCAL_CACHE}"
    python - <<PY
import hashlib, os, time
from concurrent.futures import ThreadPoolExecutor
import moxing as mox
from omegaconf import OmegaConf

os.environ.pop("DATASET_CACHE_DIR", None)          # 取配置里的 OBS 缓存目录
cfg = OmegaConf.load(r"""${DATA_CFG}""")
dst = r"""${LOCAL_CACHE}"""
os.makedirs(dst, exist_ok=True)

# 缓存文件名 = {dataset_label}_{md5(data_root)[:8]}_cache.pkl；按哈希挑出这次要用的文件
need, src_dirs = set(), set()
for sec in ("train_dataset", "test_dataset"):
    for name in cfg[sec].weights:
        b = cfg[sec][name]
        need.add(hashlib.md5(os.path.abspath(str(b.data_root)).encode()).hexdigest()[:8])
        src_dirs.add(str(OmegaConf.to_container(b, resolve=True)["cache_dir"]))
assert len(src_dirs) == 1, src_dirs
src = src_dirs.pop()
files = [f for f in mox.file.list_directory(src) if any(f.endswith(f"_{h}_cache.pkl") for h in need)]
print(f"[cache] {len(files)} files for {len(need)} data roots from {src}", flush=True)

def fetch(f):
    s, d = f"{src}/{f}", os.path.join(dst, f)
    size = mox.file.get_size(s)
    if os.path.exists(d) and os.path.getsize(d) == size:
        return f, size, 0.0, "skip"
    t = time.time()
    mox.file.copy(s, d + ".tmp")
    os.replace(d + ".tmp", d)
    return f, size, time.time() - t, "copied"

t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for f, size, dt, st in ex.map(fetch, files):
        print(f"[cache] {st:6s} {f}  {size / 1e9:.2f} GB  {dt:.0f}s", flush=True)
print(f"[cache] done in {time.time() - t0:.0f}s", flush=True)
PY
    export DATASET_CACHE_DIR="${LOCAL_CACHE}"
    echo "DATASET_CACHE_DIR=${DATASET_CACHE_DIR}"
fi

# 关键环境
export DEVICE_TYPE=ascend
export TOKENIZERS_PARALLELISM=false
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export USE_MEMARTS=0

# 多机参数（ModelArts 多机作业会设置这些变量；单机手动启动时按单机 8 卡）
MASTER_ADDR="${VC_WORKER_HOSTS:-127.0.0.1}"
MASTER_ADDR="${MASTER_ADDR%%,*}"
NNODES="${MA_NUM_HOSTS:-1}"
NODE_RANK="${VC_TASK_INDEX:-0}"
NGPUS_PER_NODE="${MA_NUM_GPUS:-8}"
NUM_PROCESSES=$((NGPUS_PER_NODE * NNODES))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"

export USE_CHECKPOINT=False
# export ASCEND_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
NAME="${NAME:-exp1023_pi3_3B_datamix_all_v9-5-sub00_filter}"

# 实验结果（checkpoint、tensorboard）上传到贵阳 efs：${EXP_ROOT}/pi3sup/outputs/${NAME}/
EXP_ROOT="${EXP_ROOT:-/efs-mi-pay-gy1/z00533370/exp}"
mkdir -p "${EXP_ROOT}/pi3sup/outputs/${NAME}"

# -----------------------------
# Auto copy latest ckpt (OBS / efs) to local output dir
# -----------------------------
# 原脚本（0604）从 ModelArts 自动同步到 OBS 的 train_url_0 里找 checkpoint：
#   obs://bucket-mi-east4-sf/z00533370/project/proj_3dgs/exp_pi3sup/pi3sup/outputs/${NAME}/outputs/${NAME}/ckpts
# 贵阳访问不到东四桶，改成贵阳桶；OBS_TRAIN_ROOT 必须与本次作业「训练输出」的 OBS 路径一致。
AUTO_RESUME="${AUTO_RESUME:-1}"
OBS_TRAIN_ROOT="${OBS_TRAIN_ROOT:-obs://bucket-471/z00533370/exp/pi3sup/outputs/${NAME}}"
OBS_CKPT_ROOT="${OBS_TRAIN_ROOT}/outputs/${NAME}/ckpts"
EFS_CKPT_ROOT="${EXP_ROOT}/pi3sup/outputs/${NAME}/ckpts"
LOCAL_CKPT_ROOT="${TRAIN_URL}/outputs/${NAME}/ckpts"
LATEST_CKPT_FILE="/tmp/latest_ckpt_${NAME}.txt"
mkdir -p "${LOCAL_CKPT_ROOT}"
rm -f "${LATEST_CKPT_FILE}"
echo "OBS_CKPT_ROOT=${OBS_CKPT_ROOT}"
echo "EFS_CKPT_ROOT=${EFS_CKPT_ROOT}"

if [ "${AUTO_RESUME}" = "1" ]; then
    echo "==> Checking latest checkpoint on OBS and efs"

    python - <<PY
import re
import moxing as mox

roots = [r"""${OBS_CKPT_ROOT}""", r"""${EFS_CKPT_ROOT}"""]
out = r"""${LATEST_CKPT_FILE}"""

latest, latest_id = None, -1
for root in roots:
    try:
        if not mox.file.exists(root):
            print(f"[AUTO_RESUME] not found: {root}")
            continue
        items = mox.file.list_directory(root)
    except Exception as e:
        print(f"[AUTO_RESUME] skip {root}: {e}")
        continue
    for item in items:
        path = item if item.startswith("obs://") else root.rstrip("/") + "/" + item.rstrip("/")
        m = re.match(r"checkpoint_(\\d+)$", path.rstrip("/").split("/")[-1])
        if m and int(m.group(1)) > latest_id:
            latest_id, latest = int(m.group(1)), path

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
        if [ ! -e "${LOCAL_CKPT}" ]; then
            copy_by_mox "${REMOTE_CKPT}" "${LOCAL_CKPT}"
        fi
        ls -lah "${LOCAL_CKPT_ROOT}" || true
    fi
fi

# -----------------------------
# Training Process（参数与 run_exp0604 相同，只有 data / name / worker_reseed 不同）
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
  data=datamix_all_v9-5-sub00 \
  ++train.worker_reseed=true \
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
  loss.test_loss.w_normalized_loss=1.0 \
  "$@"

echo "==> Training finished"

if [ "${UPLOAD:-1}" != "1" ]; then
    echo "==> UPLOAD=0, skip uploading checkpoint / tensorboard to efs"
    exit 0
fi

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
