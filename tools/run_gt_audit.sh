#!/usr/bin/env bash
# GT 体检：先冒烟测试，再正式跑 tools/gt_audit.py。在 SceneGeneration 根目录或任意目录执行均可。
#
# 用法：
#   bash tools/run_gt_audit.sh smoke        # 2 个数据集 × 3 个样本，确认环境和数据路径
#   bash tools/run_gt_audit.sh stats        # 只统计全部数据集的场景/序列/图像数，不做体检（快）
#   bash tools/run_gt_audit.sh full         # 全部数据集：规模统计 + 几何体检（前台）
#   bash tools/run_gt_audit.sh full --bg    # 同上，后台运行，总日志写到 ${OUT}.log
# 找不到 data_root 的数据集会标为 PATH_NOT_FOUND 并跳过，不影响其他数据集。
#
# 可用环境变量覆盖：
#   CONFIG=configs/data/datamix_all_v9-5.yaml
#   DATASETS="Hypersim DL3DV"     # 默认 ALL = config 里 weights 的全部数据集
#   N=30  OUT=gt_audit_out  TAU=0.03
#   PARALLEL=6                    # 同时跑几个数据集（每个数据集一个进程，日志在 ${OUT}/logs/）；1 = 串行
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

STAGE="${1:-smoke}"
BG="${2:-}"
CONFIG="${CONFIG:-configs/data/datamix_all_v9-5.yaml}"
TAU="${TAU:-0.03}"

case "${STAGE}" in
    smoke)
        DATASETS="${DATASETS:-Hypersim DL3DV}"
        N="${N:-3}"
        OUT="${OUT:-gt_audit_smoke}"
        PARALLEL="${PARALLEL:-1}"
        EXTRA=(--verbose)
        ;;
    stats)
        DATASETS="${DATASETS:-ALL}"
        N="${N:-0}"
        OUT="${OUT:-gt_audit_stats}"
        PARALLEL="${PARALLEL:-8}"
        EXTRA=(--stats-only)
        ;;
    full)
        DATASETS="${DATASETS:-ALL}"
        N="${N:-30}"
        OUT="${OUT:-gt_audit_out}"
        PARALLEL="${PARALLEL:-6}"
        EXTRA=(--raw-compare)
        ;;
    *)
        echo "usage: bash tools/run_gt_audit.sh [smoke|stats|full] [--bg]" >&2
        exit 1
        ;;
esac

if [ "${BG}" = "--bg" ]; then
    mkdir -p "${OUT}"
    DATASETS="${DATASETS}" N="${N}" OUT="${OUT}" TAU="${TAU}" CONFIG="${CONFIG}" PARALLEL="${PARALLEL}" \
        nohup bash "$0" "${STAGE}" > "${OUT}.log" 2>&1 &
    echo "==> running in background, pid=$!  log: tail -f ${OUT}.log"
    exit 0
fi

if [ "${DATASETS}" = "ALL" ]; then
    DATASETS="$(python -c "from omegaconf import OmegaConf; print(' '.join(OmegaConf.load('${CONFIG}').train_dataset.weights.keys()))")"
fi

COMMON=(--config "${CONFIG}" --n "${N}" --tau "${TAU}" --out "${OUT}" "${EXTRA[@]}")
echo "==> stage=${STAGE} config=${CONFIG} n=${N} out=${OUT} parallel=${PARALLEL}"
echo "==> datasets=${DATASETS}"
mkdir -p "${OUT}/logs"

if [ "${PARALLEL}" -le 1 ]; then
    # shellcheck disable=SC2086
    python tools/gt_audit.py "${COMMON[@]}" --datasets ${DATASETS} 2>&1 | tee "${OUT}/logs/all.log"
else
    for ds in ${DATASETS}; do
        while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do sleep 2; done
        echo "==> start ${ds}  (log: ${OUT}/logs/${ds}.log)"
        python tools/gt_audit.py "${COMMON[@]}" --datasets "${ds}" > "${OUT}/logs/${ds}.log" 2>&1 \
            && echo "==> done  ${ds}: $(grep -m1 'status=' "${OUT}/logs/${ds}.log" | sed 's/^ *//')" \
            || echo "==> FAIL  ${ds}: see ${OUT}/logs/${ds}.log" &
    done
    wait
    echo
    python tools/gt_audit.py --summarize --config "${CONFIG}" --out "${OUT}"
fi
