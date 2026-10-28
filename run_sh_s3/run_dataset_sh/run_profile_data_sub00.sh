#!/usr/bin/env bash
# 数据读取耗时分析（tools/profile_data.py）：按训练的 sub00 配置测每个数据集的单样本耗时、OBS 读取、视差重选次数，
# 估算按权重混合后每步要等多久，并做多进程并发取样测试。只用 CPU 和 OBS，不占 NPU。
#
# 用法（ModelArts 作业或 devserver，单节点即可）：
#   bash run_sh_s3/run_dataset_sh/run_profile_data_sub00.sh
#   N=10 DATASETS="MegaDepth CO3D Spring Habitat ASE" bash run_sh_s3/run_dataset_sh/run_profile_data_sub00.sh
# 可用环境变量：
#   DATA      数据配置名（默认 datamix_all_v9-5-sub00）
#   DATASETS  只测这些数据集（空格分隔，默认全部）
#   N=6       每个数据集测多少个样本
#   WORKERS=4 GPUS=128   估算用：训练时每卡 worker 数、训练卡数
#   PROCS=16 SECONDS_CONC=300   并发测试的进程数与时长（PROCS=0 跳过）
#   OUT_ROOT  结果目录（默认 /efs-mi-pay-gy1/z00533370/exp/profile_data/<时间>）
#   LOCAL_CACHE 索引缓存拷到的本地目录（默认 /cache/dataset_cache；设成空则直接读 OBS）
set -euo pipefail

# ModelArts 追加的 --key=value 作业参数（--train_url、--HCCL_* 等）这里用不到，忽略
for arg in "$@"; do
    case "${arg}" in
        --*) echo "[args] ignore ${arg}" ;;
    esac
done

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
echo "CODE_DIR=${CODE_DIR}  ($(git log --oneline -1 2>/dev/null || echo 'not a git repo'))"

# 依赖：与训练脚本相同
python - <<'PY'
try:
    import hydra_colorlog
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hydra_colorlog"])
PY

# -----------------------------
# 索引缓存拷到本地盘（每个节点一次；同名同大小的跳过）
# -----------------------------
DATA="${DATA:-datamix_all_v9-5-sub00}"
DATA_CFG="configs/data/${DATA}.yaml"
echo "DATA=${DATA}"
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

N="${N:-6}"
WORKERS="${WORKERS:-4}"
GPUS="${GPUS:-128}"
PROCS="${PROCS:-16}"
SECONDS_CONC="${SECONDS_CONC:-300}"
OUT_ROOT="${OUT_ROOT:-/efs-mi-pay-gy1/z00533370/exp/profile_data/$(date +%m%d_%H%M)}"
mkdir -p "${OUT_ROOT}"
echo "==> profile: DATA=${DATA} N=${N} WORKERS=${WORKERS} GPUS=${GPUS} PROCS=${PROCS} SECONDS=${SECONDS_CONC} OUT=${OUT_ROOT}"

# shellcheck disable=SC2086
python tools/profile_data.py --config "${DATA_CFG}" ${DATASETS:+--datasets ${DATASETS}} \
    --n "${N}" --workers "${WORKERS}" --gpus "${GPUS}" --procs "${PROCS}" --seconds "${SECONDS_CONC}" \
    --out "${OUT_ROOT}" 2>&1 | tee "${OUT_ROOT}/profile.log"

echo "==> finished: ${OUT_ROOT}/profile.log  ${OUT_ROOT}/profile_per_dataset.tsv"
