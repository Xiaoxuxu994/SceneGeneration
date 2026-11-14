#!/usr/bin/env bash
# RealEstate10K（pixelSplat chunk 格式）转成每个视频切片一个 tar（tools/re10k_to_tar.py），图像码流原样、全部帧保留。
#   默认 obs://bucket-471/dataset/re10k/{train,test}/*.torch -> obs://bucket-471/z00533370/dataset/RE10K_tar/{train,test}/<key>.tar
# train 约 4866 个 chunk（约 6.4 万切片、880 万帧、~480GB），写出的文件数 = 切片数，瓶颈在下载和解析 chunk。
# 可断点续跑：_done/<chunk>.json 已写出的 chunk 跳过；全部做完后自动合并 <split>/_index.json。只用 CPU 和 OBS。
#
# ModelArts（云道）作业（8 卡规格的 CPU 多；多节点直接用同一条命令，按 chunk 自动切分）：
#   cd /home/ma-user/modelarts/user-job-dir/SceneGeneration/; source /home/ma-user/.bashrc; bash run_sh_s3/run_utils/re10k_to_tar.sh
# 可用环境变量：SRC / OUT（默认见上）、SPLITS="train test"、PROCS=16（同时处理的 chunk 数）、THREADS=8（每个 chunk 并发写的 tar 数）、
#   LOCAL_TMP=/cache/re10k_to_tar、LIMIT=N（试跑每个 split 只做前 N 个 chunk）、MERGE_ONLY=1（只合并 _index.json）
# 日志：每行带时间戳，另存 /efs-mi-pay-gy1/z00533370/exp/logs/re10k_to_tar_node<N>_<时间>.log（目录可写时）
set -uo pipefail
for arg in "$@"; do case "${arg}" in -*) echo "[args] ignore ${arg}" ;; esac; done

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${CODE_DIR}"
export PYTHONUNBUFFERED=1
source "${CODE_DIR}/run_sh_s3/_pip_deps.sh"   # 公共依赖：缺什么装什么（见该文件）

SRC="${SRC:-obs://bucket-471/dataset/re10k}"
OUT="${OUT:-obs://bucket-471/z00533370/dataset/RE10K_tar}"
PROCS="${PROCS:-16}"
THREADS="${THREADS:-8}"
LIMIT="${LIMIT:-0}"
NODE_RANK="${VC_TASK_INDEX:-${NODE_RANK:-0}}"
NNODES="${MA_NUM_HOSTS:-${NNODES:-1}}"
LOG_DIR="${LOG_DIR:-/efs-mi-pay-gy1/z00533370/exp/logs}"
LOGFILE=/dev/null
if mkdir -p "${LOG_DIR}" 2>/dev/null && [ -w "${LOG_DIR}" ]; then
    LOGFILE="${LOG_DIR}/re10k_to_tar_node${NODE_RANK}_$(date +%m%d_%H%M).log"
fi
ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%F %T')" "${line}"; done; }

ARGS=(--src "${SRC}" --out "${OUT}" --procs "${PROCS}" --threads "${THREADS}" --shard "${NODE_RANK}/${NNODES}")
[ "${LIMIT}" -gt 0 ] && ARGS+=(--limit "${LIMIT}")
[ -n "${LOCAL_TMP:-}" ] && ARGS+=(--local-tmp "${LOCAL_TMP}")
[ "${MERGE_ONLY:-0}" = "1" ] && ARGS+=(--merge-only)
# shellcheck disable=SC2206
[ -n "${SPLITS:-}" ] && ARGS+=(--splits ${SPLITS})

echo "==> node ${NODE_RANK}/${NNODES}  ${SRC} -> ${OUT}  PROCS=${PROCS} THREADS=${THREADS} LIMIT=${LIMIT} SPLITS=${SPLITS:-train test}  log ${LOGFILE}" | ts | tee -a "${LOGFILE}"
python tools/re10k_to_tar.py "${ARGS[@]}" 2>&1 | grep -a --line-buffered -vE "^INFO:root|^\s+(stat|errorCode|errorMessage|reason|request-id|retry|func|args|kwargs)" | ts | tee -a "${LOGFILE}"
rc=${PIPESTATUS[0]}
echo "==> finished rc=${rc}" | ts | tee -a "${LOGFILE}"
exit "${rc}"
