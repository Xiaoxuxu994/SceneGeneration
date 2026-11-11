#!/usr/bin/env bash
# SpatialVID-HQ 抽帧（tools/preprocess_spatialvid.py --no-depth）：从 OBS 上的 videos / annotations tar.gz 流式读取，
# 按官方规则（每 int(fps/5) 帧取一帧）抽帧，连同相机与原始标注写到新的 OBS 目录。只用 CPU 和 OBS。
# 可断点续跑：写完的 group 有 _DONE，重跑同一条命令只补做没完成的 group。多节点时按 group 自动切分。
#
# ModelArts（云道）启动命令：
#   cd /home/ma-user/modelarts/user-job-dir/SceneGeneration/; source /home/ma-user/.bashrc; bash run_sh_s3/run_dataset_sh/run_preprocess_spatialvid.sh
# 可用环境变量（写在命令前面）：
#   OUT            默认 obs://bucket-471/z00533370/dataset/SpatialVID-HQ_rgbpose
#   MAX_SIDE=960   帧的长边（原视频 1280x720；960 约省一半存储）
#   JPEG_Q=90      JPEG 质量
#   PROCS=12       本节点同时处理的 group 数；VIDEO_WORKERS=6 每个 group 并发解码的视频数；THREADS=16 每个 group 并发写 OBS 数
#   SV_GROUPS      只做这些 group（空格分隔），默认全部 74 个
#   LIMIT_CLIPS    >0 时每个 group 只做前 N 个 clip（试跑，不写 _DONE）
#   PROGRESS_INTERVAL=600  每隔多少秒打印一次整体进度（完成的 group 数、帧数、内存），0 = 关闭
# 日志：每行带时间戳，另存 /efs-mi-pay-gy1/z00533370/exp/logs/preprocess_spatialvid_rgbpose_node<N>_<时间>.log
set -euo pipefail

for arg in "$@"; do
    case "${arg}" in
        --*) echo "[args] ignore ${arg}" ;;
    esac
done

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
source "${CODE_DIR}/run_sh_s3/_pip_deps.sh"   # 公共依赖：缺什么装什么（见该文件）

OUT="${OUT:-obs://bucket-471/z00533370/dataset/SpatialVID-HQ_rgbpose}"
MAX_SIDE="${MAX_SIDE:-960}"
JPEG_Q="${JPEG_Q:-90}"
PROCS="${PROCS:-12}"
VIDEO_WORKERS="${VIDEO_WORKERS:-6}"
THREADS="${THREADS:-16}"
LIMIT_CLIPS="${LIMIT_CLIPS:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-600}"
NODE_RANK="${VC_TASK_INDEX:-${NODE_RANK:-0}}"
NNODES="${MA_NUM_HOSTS:-${NNODES:-1}}"
LOG_DIR="${LOG_DIR:-/efs-mi-pay-gy1/z00533370/exp/logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR=.
LOG="${LOG_DIR}/preprocess_spatialvid_rgbpose_node${NODE_RANK}_$(date +%m%d_%H%M).log"
STOP="$(mktemp -u /tmp/spatialvid_progress_stop.XXXXXX)"   # 主进程结束时创建，进度循环看到后退出

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%F %T')" "${line}"; done; }

{
echo "==> CODE_DIR=${CODE_DIR} ($(git log --oneline -1 2>/dev/null || echo 'not a git repo'))"
echo "==> node ${NODE_RANK}/${NNODES}  OUT=${OUT}  MAX_SIDE=${MAX_SIDE} JPEG_Q=${JPEG_Q} PROCS=${PROCS} VIDEO_WORKERS=${VIDEO_WORKERS} THREADS=${THREADS} LIMIT_CLIPS=${LIMIT_CLIPS}"
echo "==> log ${LOG}"
} | ts | tee -a "${LOG}"

# 整体进度：完成的 group 数与帧数（读各 group 的 _DONE）、本节点内存与负载
if [ "${PROGRESS_INTERVAL}" -gt 0 ] 2>/dev/null; then
    (
        set +eu +o pipefail
        T0=$(date +%s)
        while :; do
            for _ in $(seq $(( (PROGRESS_INTERVAL + 4) / 5 ))); do
                [ -f "${STOP}" ] && exit 0
                sleep 5
            done
            python - "${OUT}" "${T0}" <<'PY' 2>/dev/null | grep -v '^INFO'
import sys, time, moxing as mox
out, t0 = sys.argv[1], float(sys.argv[2])
try:
    groups = [g.rstrip("/") for g in mox.file.list_directory(out) if g.startswith("group_")]
except Exception:
    groups = []
done, clips, frames = 0, 0, 0
for g in groups:
    p = f"{out}/{g}/_DONE"
    if mox.file.exists(p):
        done += 1
        t = mox.file.read(p).split()
        clips += int(t[0]); frames += int(t[2])
h = (time.time() - t0) / 3600
eta = f"{h / done * (74 - done):.1f}h (all nodes, 74 groups)" if done else "-"
print(f"[progress] groups done {done}/74, started {len(groups)}, clips {clips}, frames {frames}, elapsed {h:.2f}h, eta {eta}")
PY
            echo "[progress] $(free -g | awk '/^Mem:/{printf "mem used=%sG avail=%sG", $3, $7}') load=$(cut -d' ' -f1 /proc/loadavg)"
        done
    ) 2>&1 | ts | tee -a "${LOG}" &
fi

ARGS=(--no-depth --out "${OUT}" --max-side "${MAX_SIDE}" --jpeg-quality "${JPEG_Q}" --procs "${PROCS}"
      --video-workers "${VIDEO_WORKERS}" --threads "${THREADS}" --shard "${NODE_RANK}/${NNODES}")
[ "${LIMIT_CLIPS}" -gt 0 ] && ARGS+=(--limit-clips "${LIMIT_CLIPS}")
# shellcheck disable=SC2206
[ -n "${SV_GROUPS:-}" ] && ARGS+=(--groups ${SV_GROUPS})

set +e
python tools/preprocess_spatialvid.py "${ARGS[@]}" 2>&1 | grep --line-buffered -vE "^INFO|^\s+(stat|errorCode|errorMessage|reason|request-id|retry|func|args|kwargs)|UserWarning|warnings.warn" | ts | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e
touch "${STOP}"
wait
rm -f "${STOP}"
echo "==> finished rc=${rc}" | ts | tee -a "${LOG}"
exit "${rc}"
