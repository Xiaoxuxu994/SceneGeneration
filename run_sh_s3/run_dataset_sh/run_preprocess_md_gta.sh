#!/usr/bin/env bash
# MegaDepth / GTA-SfM 预处理（tools/preprocess_megadepth.py、tools/preprocess_gta_sfm.py），结果写回贵阳 OBS。
# 只用 CPU 和 OBS，单节点即可；可断点续跑（已写完 _DONE 的场景会跳过），中断后重跑同一条命令即可。
#
# 用法：
#   bash run_sh_s3/run_dataset_sh/run_preprocess_md_gta.sh                 # 两个都做
#   WHAT=megadepth bash run_sh_s3/run_dataset_sh/run_preprocess_md_gta.sh  # 只做 MegaDepth（gta = 只做 GTA）
#   SCENES="0001 0015" WHAT=megadepth bash ...                              # 先试几个场景
# 可用环境变量：
#   WHAT        both | megadepth | gta（默认 both）
#   MD_OUT      默认 obs://bucket-471/z00533370/dataset/MegaDepth_pre1024
#   GTA_OUT     默认 obs://bucket-471/z00533370/dataset/GTA-SfM_frames
#   MAX_SIDE=1024  PROCS=16  THREADS=8
#   DATA        数据配置名（默认 datamix_all_v9-5-sub00；用它的索引缓存与 path_remap）
set -euo pipefail

# ModelArts 追加的 --key=value 作业参数这里用不到，忽略
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

python - <<'PY'
try:
    import pyquaternion
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyquaternion"])
PY

WHAT="${WHAT:-both}"
DATA="${DATA:-datamix_all_v9-5-sub00}"
MD_OUT="${MD_OUT:-obs://bucket-471/z00533370/dataset/MegaDepth_pre1024}"
GTA_OUT="${GTA_OUT:-obs://bucket-471/z00533370/dataset/GTA-SfM_frames}"
MAX_SIDE="${MAX_SIDE:-1024}"
PROCS="${PROCS:-16}"
THREADS="${THREADS:-8}"
SCENE_ARGS=()
if [ -n "${SCENES:-}" ]; then
    # shellcheck disable=SC2206
    SCENE_ARGS=(--scenes ${SCENES})
fi
LOG_DIR="${LOG_DIR:-/efs-mi-pay-gy1/z00533370/exp/logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR=.
TS="$(date +%m%d_%H%M)"

rc=0
if [ "${WHAT}" = "both" ] || [ "${WHAT}" = "megadepth" ]; then
    echo "==> MegaDepth -> ${MD_OUT}  (log ${LOG_DIR}/preprocess_megadepth_${TS}.log)"
    python tools/preprocess_megadepth.py --config "configs/data/${DATA}.yaml" --out "${MD_OUT}" \
        --max-side "${MAX_SIDE}" --procs "${PROCS}" --threads "${THREADS}" ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} \
        2>&1 | tee "${LOG_DIR}/preprocess_megadepth_${TS}.log" || rc=1
fi
if [ "${WHAT}" = "both" ] || [ "${WHAT}" = "gta" ]; then
    echo "==> GTA-SfM -> ${GTA_OUT}  (log ${LOG_DIR}/preprocess_gta_${TS}.log)"
    python tools/preprocess_gta_sfm.py --config "configs/data/${DATA}.yaml" --out "${GTA_OUT}" \
        --procs "${PROCS}" --threads "${THREADS}" ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} \
        2>&1 | tee "${LOG_DIR}/preprocess_gta_${TS}.log" || rc=1
fi
echo "==> finished (rc=${rc})"
exit ${rc}
