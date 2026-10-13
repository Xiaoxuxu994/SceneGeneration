#!/usr/bin/env bash
# GT 体检：先冒烟测试，再正式跑 tools/gt_audit.py。在 SceneGeneration 根目录或任意目录执行均可。
#
# 用法：
#   bash tools/run_gt_audit.sh smoke        # 2 个数据集 × 3 个样本，确认环境和数据路径
#   bash tools/run_gt_audit.sh full         # 正式跑（前台）
#   bash tools/run_gt_audit.sh full --bg    # 正式跑（后台，日志写到 ${OUT}.log）
#
# 可用环境变量覆盖：
#   CONFIG=configs/data/datamix_all_v10-1.yaml
#   DATASETS="Hypersim DL3DV"     # 设为 ALL 则跑 config 里 weights 的全部数据集
#   N=30  OUT=gt_audit_out  TAU=0.03
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

STAGE="${1:-smoke}"
BG="${2:-}"
CONFIG="${CONFIG:-configs/data/datamix_all_v10-1.yaml}"
TAU="${TAU:-0.03}"

DEFAULT_DATASETS="Hypersim MatrixCity Scannetpp DL3DV CO3DV2 OmniObject3DNew ASE Unreal4k MVSSynth HY3DBench SAGE10K ScanNet ARKitScenes MegaDepth"

case "${STAGE}" in
    smoke)
        DATASETS="${DATASETS:-Hypersim DL3DV}"
        N="${N:-3}"
        OUT="${OUT:-gt_audit_smoke}"
        EXTRA=(--verbose)
        ;;
    full)
        DATASETS="${DATASETS:-${DEFAULT_DATASETS}}"
        N="${N:-30}"
        OUT="${OUT:-gt_audit_out}"
        EXTRA=(--raw-compare)
        ;;
    *)
        echo "usage: bash tools/run_gt_audit.sh [smoke|full] [--bg]" >&2
        exit 1
        ;;
esac

ARGS=(--config "${CONFIG}" --n "${N}" --tau "${TAU}" --out "${OUT}" "${EXTRA[@]}")
if [ "${DATASETS}" != "ALL" ]; then
    # shellcheck disable=SC2206
    ARGS+=(--datasets ${DATASETS})
fi

echo "==> stage=${STAGE} config=${CONFIG} n=${N} out=${OUT}"
echo "==> datasets=${DATASETS}"

if [ "${BG}" = "--bg" ]; then
    nohup python tools/gt_audit.py "${ARGS[@]}" > "${OUT}.log" 2>&1 &
    echo "==> running in background, pid=$!  log: tail -f ${OUT}.log"
else
    python tools/gt_audit.py "${ARGS[@]}" 2>&1 | tee "${OUT}.log"
fi
