#!/usr/bin/env bash
# 把 EFS 上的 Pi3 3B checkpoint 和对应训练 log 用 moxing 备份到 OBS，OBS 上保持 EFS 的目录结构：
#   ckpt  /efs-m3ga/c50041401/exp/pi3sup/outputs/<exp>/ckpts/<ckpt>  ->  ${OBS_ROOT}/<exp>/ckpts/<ckpt>
#   log   /efs-m3ga/c50041401/exp/pi3sup/outputs/<exp>/<exp>/        ->  ${OBS_ROOT}/<exp>/<exp>/
# 实验名这一层不能省：有 3 个 model_ckpt3、2 个 model_ckpt39，省掉会互相覆盖。
#
# ModelArts（云道）作业，只用 1 个节点（多节点会每个节点都重复上传一遍），作业要挂载 /efs-m3ga 且能读 c50041401 的目录：
#   cd /home/ma-user/modelarts/user-job-dir/SceneGeneration/; source /home/ma-user/.bashrc; bash run_sh_s3/run_utils/mox_ckpt_bak.sh
# 用法：
#   bash run_sh_s3/run_utils/mox_ckpt_bak.sh                          # 默认的 6 个 checkpoint + 各自的 log
#   bash run_sh_s3/run_utils/mox_ckpt_bak.sh <efs_ckpt_dir> [...]     # 只备份这些
#   CKPTS="<dir> <dir>" bash run_sh_s3/run_utils/mox_ckpt_bak.sh      # 同上（ModelArts 作业不便加位置参数时）
#   DRY=1   只打印每项的源大小、文件数和目标路径，不拷贝（3B 的 ckpt 可能几十 GB，先跑一次确认总量）
#   LOG=0   只备份 ckpt，不备份 log
#   FORCE=1 OBS 上已有且大小一致也重新拷
# 拷贝：目录用 mox.file.copy_parallel，单个文件用 mox.file.copy；拷完比较 EFS 与 OBS 的总字节数，不一致算失败。
# 可续传：OBS 上已存在且总大小一致的直接跳过，作业中断后重新提交即从断点继续。源目录只读，不写不删。
# 以 - 开头的参数忽略（ModelArts 会在启动命令后追加 --HCCL_CONNECT_TIMEOUT=7200 等超参数）。
# 最后打印汇总，有失败时退出码为 1。日志每行带时间戳；LOG_DIR 可写时另存一份（默认 /efs-mi-pay-gy1/z00533370/exp/logs）。
set -uo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
[ -f "${CODE_DIR}/run_sh_s3/_pip_deps.sh" ] && source "${CODE_DIR}/run_sh_s3/_pip_deps.sh"

OBS_ROOT="${OBS_ROOT:-obs://bucket-mi-east4-sf/z00533370/ckpt_bak}"
DRY="${DRY:-0}"
FORCE="${FORCE:-0}"
LOG="${LOG:-1}"
LOG_DIR="${LOG_DIR:-/efs-mi-pay-gy1/z00533370/exp/logs}"
LOGFILE=/dev/null
if mkdir -p "${LOG_DIR}" 2>/dev/null && [ -w "${LOG_DIR}" ]; then
    LOGFILE="${LOG_DIR}/mox_ckpt_bak_$(date +%m%d_%H%M%S).log"
fi

SRCS=()
for arg in ${CKPTS:-} "$@"; do
    case "${arg}" in -*) echo "[INFO] ignoring argument ${arg}" ;; *) SRCS+=("${arg}") ;; esac
done
if [ "${#SRCS[@]}" -eq 0 ]; then
    SRCS=(
        /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt79
        /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0521_pi3_3B_datamix_all_v9-1_lowres_v8-m8_randres_stage3_preexp02/ckpts/model_ckpt3
        /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0528_pi3_3B_datamix_all_v9-2_lowres_v8-m8_randres_stage4_10k/ckpts/model_ckpt3
        /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0529_pi3_3B_datamix_all_v9-5_lowres_v8-m8_randres_stage5_100k/ckpts/model_ckpt39
        /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0604_pi3_3B_datamix_all_v9-5_randresNew_scratch_uploss_pinmemTrue/ckpts/model_ckpt78
        /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0612_pi3_3B_datamix_all_v9-5_randresSet_stage2_localnormalloss_pinmemTrue/ckpts/model_ckpt39
    )
fi

RCFILE="$(mktemp /tmp/mox_ckpt_bak_rc.XXXXXX)"   # 花括号块在管道里（子 shell）跑，用文件把退出码带出来
echo 1 > "${RCFILE}"
ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%F %T')" "${line}"; done; }

# mox_bak <src> <dst> <dry> <force>：拷一个 checkpoint 或 log（目录或单个文件），再比较总字节数
mox_bak () {
    python - "$@" <<'PY'
import os, sys, time
import moxing as mox
src, dst, dry, force = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"), sys.argv[3] == "1", sys.argv[4] == "1"

def local_size(p):
    if os.path.isfile(p):
        return os.path.getsize(p), 1
    total = n = 0
    for root, _, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f)); n += 1
    return total, n

def obs_size(p):
    """OBS 上的总字节数；不存在返回 None。get_size(recursive=True) 不可用时，递归列文件逐个求和。"""
    if not mox.file.exists(p):
        return None
    if not mox.file.is_directory(p):
        return mox.file.get_size(p)
    try:
        return mox.file.get_size(p, recursive=True)
    except Exception:
        total = 0
        for f in mox.file.list_directory(p, recursive=True):   # 可能同时列出子目录项，只累加文件
            q = f"{p}/{f.rstrip('/')}"
            if f.endswith("/") or mox.file.is_directory(q):
                continue
            total += mox.file.get_size(q)
        return total

if not os.path.exists(src):
    print(f"[mox] source not found: {src}"); sys.exit(2)
size, n = local_size(src)
print(f"[mox] source {size / 2**30:.2f} GiB, {n} files")
have = obs_size(dst)
if have == size and not force:
    print(f"[mox] already on OBS with the same size, skip: {dst}"); sys.exit(0)
if have is not None:
    print(f"[mox] OBS has {have / 2**30:.2f} GiB (differs or FORCE=1), copying again")
if dry:
    print(f"[mox] DRY: would copy -> {dst}"); sys.exit(0)
t = time.time()
if os.path.isdir(src):
    mox.file.copy_parallel(src, dst)
else:
    mox.file.copy(src, dst)
dt = time.time() - t
have = obs_size(dst)
if have != size:
    print(f"[mox] SIZE MISMATCH after copy: local {size} B, OBS {have} B"); sys.exit(3)
print(f"[mox] done in {dt / 60:.1f} min ({size / 2**20 / max(dt, 1e-6):.0f} MiB/s), size verified: {dst}")
PY
}

{
echo "==> OBS_ROOT=${OBS_ROOT}  DRY=${DRY}  FORCE=${FORCE}  LOG=${LOG}  items: ${#SRCS[@]}"
echo "==> log ${LOGFILE}"
FAILED=()
for SRC in "${SRCS[@]}"; do
    SRC="${SRC%/}"
    CKPT="$(basename "${SRC}")"                  # model_ckpt79
    EXP_DIR="$(dirname "$(dirname "${SRC}")")"   # .../outputs/<exp>（ckpts/ 的上一级）
    EXP="$(basename "${EXP_DIR}")"
    echo "================ ${EXP} ================"

    echo "==> ckpt ${SRC}"
    echo "--> ${OBS_ROOT}/${EXP}/ckpts/${CKPT}"
    mox_bak "${SRC}" "${OBS_ROOT}/${EXP}/ckpts/${CKPT}" "${DRY}" "${FORCE}" || FAILED+=("${EXP}/ckpts/${CKPT}")

    if [ "${LOG}" = "1" ]; then
        echo "==> log  ${EXP_DIR}/${EXP}"
        echo "--> ${OBS_ROOT}/${EXP}/${EXP}"
        mox_bak "${EXP_DIR}/${EXP}" "${OBS_ROOT}/${EXP}/${EXP}" "${DRY}" "${FORCE}" || FAILED+=("${EXP}/${EXP} (log)")
    fi
done

echo "================ summary ================"
echo "OBS: ${OBS_ROOT}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[WARN] failed: ${FAILED[*]}"
    echo "==> finished rc=1"
else
    echo "==> finished rc=0"
    echo 0 > "${RCFILE}"
fi
} 2>&1 | grep --line-buffered -vE "^INFO:root|^\s+(stat|errorCode|errorMessage|reason|request-id|retry|func|args|kwargs)" | ts | tee -a "${LOGFILE}"
RC="$(cat "${RCFILE}")"
rm -f "${RCFILE}"
exit "${RC}"
