#!/usr/bin/env bash
# 标签质量评估（M1–M7，有真值时加 G1）：tools/quality_eval.py 的一键脚本。指标定义见报告"质量指标定义"一页。
#
# 用法：
#   bash tools/run_quality_eval.sh test           # 单元测试（只需 numpy），确认环境
#   bash tools/run_quality_eval.sh smoke          # 2 个数据集 × 3 个样本，确认数据路径
#   bash tools/run_quality_eval.sh full [--bg]    # 全部数据集的真值质量（含 depth_mask / 98 分位两条规则的 M7）
#   ROOT=/path/to/pseudo bash tools/run_quality_eval.sh pseudo [--bg]   # 伪标签目录（<ROOT>/<序列>/*.npz）
#   ROOT=/path/to/pseudo_with_gt bash tools/run_quality_eval.sh calib   # 用带 gt_depth 的序列校准阈值 → calib.json
# 找不到 data_root 的数据集会标为 PATH_NOT_FOUND 并跳过，不影响其他数据集。
#
# 可用环境变量覆盖：
#   CONFIG=configs/data/datamix_all_v9-5.yaml
#   DATASETS="Hypersim DL3DV"     # 默认 ALL = config 里 weights 的全部数据集
#   N=30  FRAMES=8  OUT=quality_out  STRIDE=1
#   PARALLEL=6                    # full：同时跑几个数据集（日志在 ${OUT}/logs/）；1 = 串行
#   THRESHOLDS=calib.json         # calib 的输出，pseudo / full 都可用
#   CONF_THR=0.5  FRAME_STEP=1  SAVE_MASKS=1      # pseudo 专用
#   TARGET_ABSREL=0.03  HOLDOUT=0.3               # calib 专用
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

STAGE="${1:-smoke}"
BG="${2:-}"
CONFIG="${CONFIG:-configs/data/datamix_all_v9-5.yaml}"
FRAMES="${FRAMES:-8}"
STRIDE="${STRIDE:-1}"
COMMON=(--frames "${FRAMES}" --stride "${STRIDE}")
[ -n "${THRESHOLDS:-}" ] && COMMON+=(--thresholds "${THRESHOLDS}")

run_bg() {  # run_bg <log> <cmd...>：--bg 时后台运行，否则前台并 tee 到日志
    local log="$1"; shift
    mkdir -p "$(dirname "${log}")"
    if [ "${BG}" = "--bg" ]; then
        nohup "$@" > "${log}" 2>&1 &
        echo "==> running in background, pid=$!  log: tail -f ${log}"
    else
        "$@" 2>&1 | tee "${log}"
    fi
}

case "${STAGE}" in
    test)
        python tests/test_quality_metrics.py
        exit 0
        ;;
    smoke)
        DATASETS="${DATASETS:-Hypersim DL3DV}"
        N="${N:-3}"
        OUT="${OUT:-quality_smoke}"
        PARALLEL="${PARALLEL:-1}"
        EXTRA=(--verbose)
        ;;
    full)
        DATASETS="${DATASETS:-ALL}"
        N="${N:-30}"
        OUT="${OUT:-quality_out}"
        PARALLEL="${PARALLEL:-6}"
        EXTRA=(--raw-compare)
        ;;
    pseudo)
        : "${ROOT:?需要 ROOT=<伪标签目录>}"
        OUT="${OUT:-quality_pseudo}"
        ARGS=(pseudo --root "${ROOT}" --out "${OUT}" --frame-step "${FRAME_STEP:-1}" "${COMMON[@]}")
        [ -n "${CONF_THR:-}" ] && ARGS+=(--conf-thr "${CONF_THR}")
        [ "${SAVE_MASKS:-0}" = "1" ] && ARGS+=(--save-masks)
        run_bg "${OUT}/run.log" python tools/quality_eval.py "${ARGS[@]}"
        exit 0
        ;;
    calib)
        : "${ROOT:?需要 ROOT=<带 gt_depth 的伪标签目录>}"
        OUT="${OUT:-calib.json}"
        run_bg "${OUT%.json}.log" python tools/calibrate_thresholds.py --root "${ROOT}" --frames "${FRAMES}" \
            --stride "${STRIDE}" --target-absrel "${TARGET_ABSREL:-0.03}" --holdout-frac "${HOLDOUT:-0.3}" --out "${OUT}"
        exit 0
        ;;
    *)
        echo "usage: bash tools/run_quality_eval.sh [test|smoke|full|pseudo|calib] [--bg]" >&2
        exit 1
        ;;
esac

# ---- smoke / full：训练数据集 ----
if [ "${BG}" = "--bg" ]; then
    mkdir -p "${OUT}"
    DATASETS="${DATASETS}" N="${N}" OUT="${OUT}" CONFIG="${CONFIG}" PARALLEL="${PARALLEL}" FRAMES="${FRAMES}" \
        STRIDE="${STRIDE}" THRESHOLDS="${THRESHOLDS:-}" nohup bash "$0" "${STAGE}" > "${OUT}.log" 2>&1 &
    echo "==> running in background, pid=$!  log: tail -f ${OUT}.log"
    exit 0
fi

if [ "${DATASETS}" = "ALL" ]; then
    DATASETS="$(python -c "from omegaconf import OmegaConf; print(' '.join(OmegaConf.load('${CONFIG}').train_dataset.weights.keys()))")"
fi

DS_ARGS=(dataset --config "${CONFIG}" --n "${N}" --out "${OUT}" "${COMMON[@]}" "${EXTRA[@]}")
echo "==> stage=${STAGE} config=${CONFIG} n=${N} frames=${FRAMES} out=${OUT} parallel=${PARALLEL}"
echo "==> datasets=${DATASETS}"
mkdir -p "${OUT}/logs"

if [ "${PARALLEL}" -le 1 ]; then
    # shellcheck disable=SC2086
    python tools/quality_eval.py "${DS_ARGS[@]}" --datasets ${DATASETS} 2>&1 | tee "${OUT}/logs/all.log"
else
    for ds in ${DATASETS}; do
        while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do sleep 2; done
        echo "==> start ${ds}  (log: ${OUT}/logs/${ds}.log)"
        python tools/quality_eval.py "${DS_ARGS[@]}" --no-summary --datasets "${ds}" > "${OUT}/logs/${ds}.log" 2>&1 \
            && echo "==> done  ${ds}: $(grep -m1 'grade=' "${OUT}/logs/${ds}.log" | sed 's/^ *//')" \
            || echo "==> FAIL  ${ds}: see ${OUT}/logs/${ds}.log" &
    done
    wait
    echo
    # shellcheck disable=SC2086
    python tools/quality_eval.py summarize --out "${OUT}" --datasets ${DATASETS}
fi
