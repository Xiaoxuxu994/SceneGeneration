#!/usr/bin/env bash
# DL3DV 官方逐场景 zip 解压回 OBS（tools/unzip_dl3dv.py），内容原样不改，transforms.json 原样保留。
#   默认 obs://bucket-471/dataset/DL3DV-ALL-960P -> obs://bucket-471/z00533370/dataset/DL3DV-ALL-960P_unzip
# 约 9417 个场景、310 万个文件；OBS 小文件写入单节点约 300 个/s 封顶，单节点约 3 小时，开 N 个节点按场景自动切分。
# 可断点续跑：transforms.json 已写出的场景跳过，重新提交同一个作业即可。只用 CPU 和 OBS。
#
# ModelArts（云道）作业（8 卡规格的 CPU 多；多节点直接用同一条命令）：
#   cd /home/ma-user/modelarts/user-job-dir/SceneGeneration/; source /home/ma-user/.bashrc; bash run_sh_s3/run_dataset_sh/run_unzip_dl3dv.sh
# 可用环境变量：SRC / OUT（默认见上）、PROCS=8（同时解压的场景数）、THREADS=16（每个场景并发写的文件数）、
#   LIMIT=N（试跑只做前 N 个场景）、BATCHES="1K 2K"（只做这些批次）
# 日志：每行带时间戳，另存 /efs-mi-pay-gy1/z00533370/exp/logs/unzip_dl3dv_node<N>_<时间>.log（目录可写时）
set -uo pipefail
for arg in "$@"; do case "${arg}" in -*) echo "[args] ignore ${arg}" ;; esac; done

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${CODE_DIR}"
export PYTHONUNBUFFERED=1
source "${CODE_DIR}/run_sh_s3/_pip_deps.sh"   # 公共依赖：缺什么装什么（见该文件）

SRC="${SRC:-obs://bucket-471/dataset/DL3DV-ALL-960P}"
OUT="${OUT:-obs://bucket-471/z00533370/dataset/DL3DV-ALL-960P_unzip}"
PROCS="${PROCS:-8}"
THREADS="${THREADS:-16}"
LIMIT="${LIMIT:-0}"
NODE_RANK="${VC_TASK_INDEX:-${NODE_RANK:-0}}"
NNODES="${MA_NUM_HOSTS:-${NNODES:-1}}"
LOG_DIR="${LOG_DIR:-/efs-mi-pay-gy1/z00533370/exp/logs}"
LOGFILE=/dev/null
if mkdir -p "${LOG_DIR}" 2>/dev/null && [ -w "${LOG_DIR}" ]; then
    LOGFILE="${LOG_DIR}/unzip_dl3dv_node${NODE_RANK}_$(date +%m%d_%H%M).log"
fi
ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%F %T')" "${line}"; done; }

ARGS=(--src "${SRC}" --out "${OUT}" --procs "${PROCS}" --threads "${THREADS}" --shard "${NODE_RANK}/${NNODES}")
[ "${LIMIT}" -gt 0 ] && ARGS+=(--limit "${LIMIT}")
# shellcheck disable=SC2206
[ -n "${BATCHES:-}" ] && ARGS+=(--batches ${BATCHES})

echo "==> node ${NODE_RANK}/${NNODES}  ${SRC} -> ${OUT}  PROCS=${PROCS} THREADS=${THREADS} LIMIT=${LIMIT}  log ${LOGFILE}" | ts | tee -a "${LOGFILE}"
python tools/unzip_dl3dv.py "${ARGS[@]}" 2>&1 | grep -a --line-buffered -vE "^INFO:root|^\s+(stat|errorCode|errorMessage|reason|request-id|retry|func|args|kwargs)" | ts | tee -a "${LOGFILE}"
rc=${PIPESTATUS[0]}
echo "==> finished rc=${rc}" | ts | tee -a "${LOGFILE}"
exit "${rc}"
