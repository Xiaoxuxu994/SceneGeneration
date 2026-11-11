#!/usr/bin/env bash
# 把 ModelArts 训练作业的 checkpoint + tensorboard 日志从 OBS 下载到 efs（moxing），按实验名通用。
# 路径与 run_datamix_all/_train_pi3_obs_common.sh 一致：
#   OBS  ${OBS_ROOT}/<NAME>/outputs/<NAME>/{ckpts/<ckpt>, <NAME>}
#   EFS  ${EFS_ROOT}/<NAME>/{ckpts/<ckpt>, <NAME>}          （与训练结束时自动上传 efs 的目录相同）
#
# 用法（在能用 moxing 的机器上：ModelArts 开发机 / 作业）：
#   bash run_sh_s3/run_utils/mox_download.sh exp1102_pi3_3B_datamix_all_v9-5-sub01_filter          # 最新 ckpt + 日志
#   bash run_sh_s3/run_utils/mox_download.sh <name1> <name2> ...                                   # 多个实验
#   RUNS="<name1> <name2>" bash run_sh_s3/run_utils/mox_download.sh                                # ModelArts 作业（不便加位置参数时）
#   CKPT=checkpoint_54 bash run_sh_s3/run_utils/mox_download.sh <name>    # 指定 checkpoint（epoch 从 0 计：55 个 epoch = checkpoint_54）
#   CKPT=best_model    bash run_sh_s3/run_utils/mox_download.sh <name>    # 验证集最优
#   CKPT=list          bash run_sh_s3/run_utils/mox_download.sh <name>    # 只列出有哪些 checkpoint 及大小，不下载
#   CKPT=none          bash run_sh_s3/run_utils/mox_download.sh <name>    # 只要日志
#   CKPT=all           bash run_sh_s3/run_utils/mox_download.sh <name>    # 全部 checkpoint（很大：DeepSpeed 优化器状态）
#   LOG=0              bash run_sh_s3/run_utils/mox_download.sh <name>    # 只要 checkpoint
# 默认先合并再下载（MERGE=1）：checkpoint 先拷到本地临时盘 LOCAL_TMP，用 checkpoint 自带的 zero_to_fp32.py 合成 fp32 模型，
#   直接写到 efs 的 ckpts/model_ckpt_<N>（与 checkpoint_<N> 编号一致；新版 DeepSpeed 是目录，旧版是 .bin），
#   再删掉临时盘上的完整 checkpoint。KEEP_FULL=1 时完整的 DeepSpeed checkpoint（含优化器状态，用于续训）也下载到 efs。
#   MERGE=0：不合并，按原样下载完整 checkpoint（旧行为）。
# 可用环境变量：OBS_ROOT（默认 obs://bucket-471/z00533370/exp/pi3sup/outputs）、EFS_ROOT（默认 /efs-mi-pay-gy1/z00533370/exp/pi3sup/outputs）
# 日志：每行带时间戳，另存 /efs-mi-pay-gy1/z00533370/exp/logs/mox_download_<时间>.log
set -uo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
[ -f "${CODE_DIR}/run_sh_s3/_pip_deps.sh" ] && source "${CODE_DIR}/run_sh_s3/_pip_deps.sh"

OBS_ROOT="${OBS_ROOT:-obs://bucket-471/z00533370/exp/pi3sup/outputs}"
EFS_ROOT="${EFS_ROOT:-/efs-mi-pay-gy1/z00533370/exp/pi3sup/outputs}"
CKPT="${CKPT:-latest}"
LOG="${LOG:-1}"
MERGE="${MERGE:-1}"
KEEP_FULL="${KEEP_FULL:-0}"
LOCAL_TMP="${LOCAL_TMP:-/cache/mox_download_tmp}"
mkdir -p "${LOCAL_TMP}" 2>/dev/null || LOCAL_TMP="${EFS_ROOT}/.mox_download_tmp"
source "${CODE_DIR}/run_sh_s3/_merge_ckpt.sh"
LOG_DIR="${LOG_DIR:-/efs-mi-pay-gy1/z00533370/exp/logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR=.
LOGFILE="${LOG_DIR}/mox_download_$(date +%m%d_%H%M%S).log"

# 实验名：RUNS="a b ..." 或位置参数；以 - 开头的参数忽略（ModelArts 作业会把 --HCCL_* 等超参追加到启动命令后面）
NAMES=()
for arg in ${RUNS:-} "$@"; do
    case "${arg}" in -*) echo "[INFO] ignoring argument ${arg}" ;; *) NAMES+=("${arg}") ;; esac
done
if [ "${#NAMES[@]}" -eq 0 ]; then
    echo "[ERROR] 没有给实验名。用法：bash $0 <实验名> [更多实验名]，或 RUNS=\"<实验名> ...\" bash $0"
    echo "        OBS 上已有的实验：${OBS_ROOT}/"
    exit 2
fi

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%F %T')" "${line}"; done; }

# mox_py <mode> <src> [dst]
#   latest：打印 src 下最新的 checkpoint_N；all：打印全部 checkpoint_N；list：打印每个 checkpoint 的文件数与大小；copy：src -> dst
mox_py () {
    python - "$@" <<'PY'
import re, sys, time
import moxing as mox
mode, src = sys.argv[1], sys.argv[2].rstrip("/")
if not mox.file.exists(src):
    print(f"[mox] not found: {src}"); sys.exit(2)
if mode == "copy":
    dst = sys.argv[3]
    t = time.time()
    mox.file.copy_parallel(src, dst)
    print(f"[mox] copy done in {time.time() - t:.0f}s: {src} -> {dst}")
    sys.exit(0)
names = [x.rstrip("/").split("/")[-1] for x in mox.file.list_directory(src)]
ckpts = sorted((int(m.group(1)), n) for n in names for m in [re.match(r"checkpoint_(\d+)$", n)] if m)
if mode == "list":
    for n in [n for _, n in ckpts] + [n for n in names if not re.match(r"checkpoint_\d+$", n)]:
        d = f"{src}/{n}"
        files = [f for f in mox.file.list_directory(d, recursive=True) if not mox.file.is_directory(f"{d}/{f}")] if mox.file.is_directory(d) else []
        size = sum(mox.file.get_size(f"{d}/{f}") for f in files)
        print(f"  {n:20s} {len(files):5d} files {size / 1e9:8.1f} GB")
    sys.exit(0)
if not ckpts:
    sys.exit(3)
print(ckpts[-1][1] if mode == "latest" else " ".join(n for _, n in ckpts))
PY
}

copy_by_mox () {
    local SRC="$1" DST="$2"
    mkdir -p "$(dirname "${DST}")"
    echo "==> mox copy: ${SRC} -> ${DST}"
    mox_py copy "${SRC}" "${DST}"
}

{
echo "==> OBS_ROOT=${OBS_ROOT}  EFS_ROOT=${EFS_ROOT}  CKPT=${CKPT}  LOG=${LOG}  MERGE=${MERGE}  KEEP_FULL=${KEEP_FULL}  LOCAL_TMP=${LOCAL_TMP}  runs: ${NAMES[*]}"
echo "==> log ${LOGFILE}"
FAILED=()
for NAME in "${NAMES[@]}"; do
    echo "================ ${NAME} ================"
    OBS_RUN="${OBS_ROOT}/${NAME}/outputs/${NAME}"
    EFS_RUN="${EFS_ROOT}/${NAME}"
    ok=1

    # 1) checkpoint(s)
    case "${CKPT}" in
        none) CKPT_LIST=() ;;
        list)
            mox_py list "${OBS_RUN}/ckpts" || { echo "[WARN] no ckpts under ${OBS_RUN}/ckpts"; ok=0; }
            CKPT_LIST=() ;;
        latest)
            c="$(mox_py latest "${OBS_RUN}/ckpts" 2>/dev/null | grep -v '^INFO' | tail -1)"
            if [[ "${c}" == checkpoint_* ]]; then CKPT_LIST=("${c}"); else echo "[WARN] no checkpoint_N under ${OBS_RUN}/ckpts"; CKPT_LIST=(); ok=0; fi ;;
        all)
            read -r -a CKPT_LIST <<< "$(mox_py all "${OBS_RUN}/ckpts" 2>/dev/null | grep -v '^INFO' | tail -1)"
            [ "${#CKPT_LIST[@]}" -gt 0 ] || { echo "[WARN] no checkpoint_N under ${OBS_RUN}/ckpts"; ok=0; } ;;
        *) CKPT_LIST=("${CKPT}") ;;
    esac
    for c in ${CKPT_LIST[@]+"${CKPT_LIST[@]}"}; do
        echo "==> checkpoint: ${c}"
        if [ "${MERGE}" = "1" ] && [[ "${c}" == checkpoint_* || "${c}" == best_model ]]; then
            TMP_CK="${LOCAL_TMP}/${NAME}/${c}"
            if [ "${KEEP_FULL}" = "1" ]; then
                copy_by_mox "${OBS_RUN}/ckpts/${c}" "${EFS_RUN}/ckpts/${c}" || ok=0
                TMP_CK="${EFS_RUN}/ckpts/${c}"
            else
                copy_by_mox "${OBS_RUN}/ckpts/${c}" "${TMP_CK}" || ok=0
            fi
            merge_zero_ckpt "${TMP_CK}" "${EFS_RUN}/ckpts" || ok=0
            [ "${KEEP_FULL}" = "1" ] || rm -rf "${LOCAL_TMP:?}/${NAME}/${c}"
        else
            copy_by_mox "${OBS_RUN}/ckpts/${c}" "${EFS_RUN}/ckpts/${c}" || ok=0
        fi
    done

    # 2) tensorboard / 文本日志
    if [ "${LOG}" = "1" ] && [ "${CKPT}" != "list" ]; then
        echo "==> log"
        copy_by_mox "${OBS_RUN}/${NAME}" "${EFS_RUN}/${NAME}" || ok=0
    fi

    [ "${ok}" = "1" ] || FAILED+=("${NAME}")
done

echo "================ summary ================"
echo "EFS: ${EFS_ROOT}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[WARN] incomplete: ${FAILED[*]}"
    echo "==> finished rc=1"
else
    echo "==> finished rc=0"
fi
} 2>&1 | grep --line-buffered -vE "^INFO:root|^\s+(stat|errorCode|errorMessage|reason|request-id|retry|func|args|kwargs)" | ts | tee -a "${LOGFILE}"
grep -q "finished rc=0" "${LOGFILE}"
