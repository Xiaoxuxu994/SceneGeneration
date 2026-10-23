#!/usr/bin/env bash
# Final evaluation of the exp1022 runs (and the official Pi3) on the fixed ScanNet++ held-out clips.
# Runs on the devserver (EFS + 8 NPUs): build the benchmark once -> infer every model (8 processes share the
# clips) -> score. Re-running skips finished clips; FORCE=1 re-infers the selected models.
#
#   bash tools/eval_bench/run_eval.sh                          # official Pi3 + exp1022_00..04, latest checkpoints
#   KSWEEP=1 bash tools/eval_bench/run_eval.sh                 # also loop models at other inference K
#   ONLY="exp1022_00 exp1022_03" bash tools/eval_bench/run_eval.sh
#   CKPT=best_model bash tools/eval_bench/run_eval.sh          # checkpoint_N | best_model | latest (default)
#   N_CLIPS=500 BENCH=/efs-m3ga/z00533370/eval_bench/scannetpp_val8_500 bash tools/eval_bench/run_eval.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
export PYTHONPATH="${REPO}"
BENCH="${BENCH:-/efs-m3ga/z00533370/eval_bench/scannetpp_val8}"
RUN_ROOT="${RUN_ROOT:-/efs-m3ga/z00533370/exp/pi3sup/outputs}"          # where mox_download.sh puts checkpoints
PI3_WEIGHTS="${PI3_WEIGHTS:-/efs-m3ga/z00533370/checkpoints/yyfz233_Pi3/model.safetensors}"
CKPT="${CKPT:-latest}"
NPUS="${NPUS:-0,1,2,3,4,5,6,7}"
N_CLIPS="${N_CLIPS:-}"
IFS=',' read -r -a DEVS <<< "${NPUS}"

# name | model config | run directory | inference K (empty = loop_k_infer of the config)
MODELS=(
    "exp1022_00|pi3_dinov3s_w384|exp1022_00_scannetpp_base_dec384x36_k1_head2|"
    "exp1022_01|pi3_dinov3s_loop_w384_L1_k14to18|exp1022_01_scannetpp_loop384_L1_k14to18_head2|"
    "exp1022_02|pi3_dinov3s_loop_w384_L2_k5to9|exp1022_02_scannetpp_loop384_L2_k5to9_head2|"
    "exp1022_03|pi3_dinov3s_loop_w384_L6_k2to3|exp1022_03_scannetpp_loop384_L6_k2to3_head2|"
    "exp1022_04|pi3_dinov3s_loop_w384_L1_k1to18|exp1022_04_scannetpp_loop384_L1_k1to18_head2|"
    "exp1022_05|pi3_dinov3s_w384x12|exp1022_05_scannetpp_base_dec384x12_k1_head2|"
)
if [ "${KSWEEP:-0}" = "1" ]; then   # fewer loop steps than trained for: accuracy vs compute
    MODELS+=(
        "exp1022_01_K14|pi3_dinov3s_loop_w384_L1_k14to18|exp1022_01_scannetpp_loop384_L1_k14to18_head2|14"
        "exp1022_01_K16|pi3_dinov3s_loop_w384_L1_k14to18|exp1022_01_scannetpp_loop384_L1_k14to18_head2|16"
        "exp1022_02_K5|pi3_dinov3s_loop_w384_L2_k5to9|exp1022_02_scannetpp_loop384_L2_k5to9_head2|5"
        "exp1022_02_K7|pi3_dinov3s_loop_w384_L2_k5to9|exp1022_02_scannetpp_loop384_L2_k5to9_head2|7"
        "exp1022_03_K2|pi3_dinov3s_loop_w384_L6_k2to3|exp1022_03_scannetpp_loop384_L6_k2to3_head2|2"
        "exp1022_04_K6|pi3_dinov3s_loop_w384_L1_k1to18|exp1022_04_scannetpp_loop384_L1_k1to18_head2|6"
        "exp1022_04_K12|pi3_dinov3s_loop_w384_L1_k1to18|exp1022_04_scannetpp_loop384_L1_k1to18_head2|12"
    )
fi

selected () { [ -z "${ONLY:-}" ] && return 0; for o in ${ONLY}; do [ "$1" = "$o" ] && return 0; done; return 1; }

resolve_ckpt () {   # <run dir> -> checkpoint directory
    local d="${RUN_ROOT}/$1/ckpts"
    if [ "${CKPT}" = "latest" ]; then
        ls -d "${d}"/checkpoint_* 2>/dev/null | sort -V | tail -1
    else
        [ -d "${d}/${CKPT}" ] && echo "${d}/${CKPT}"
    fi
}

run_infer () {   # <name> <infer args...>: one process per NPU, clips are claimed atomically
    local name="$1"; shift
    [ "${FORCE:-0}" = "1" ] && rm -rf "${BENCH:?}/${name}"
    echo "==> infer ${name} on NPUs ${NPUS}"
    local pids=() i=0
    mkdir -p "${BENCH}/logs"
    for dev in "${DEVS[@]}"; do
        ASCEND_RT_VISIBLE_DEVICES="${dev}" CUDA_VISIBLE_DEVICES="${dev}" \
            python tools/eval_bench/infer.py --bench "${BENCH}" --name "${name}" "$@" \
            > "${BENCH}/logs/infer_${name}_${i}.log" 2>&1 &
        pids+=($!); i=$((i + 1))
    done
    local fail=0
    for p in "${pids[@]}"; do wait "${p}" || fail=1; done
    local n_ok n_all
    n_ok=$(grep -c '"ok": true' "${BENCH}/${name}/status.jsonl" 2>/dev/null || true)
    n_all=$(python -c "import json;print(len(json.load(open('${BENCH}/clips.json'))))")
    echo "    ${name}: ${n_ok}/${n_all} clips ok$( [ "${fail}" = 1 ] && echo ', a process failed: see '"${BENCH}"'/logs/infer_'"${name}"'_*.log')"
}

# 1) benchmark (once)
if [ ! -f "${BENCH}/clips.json" ]; then
    echo "==> build ${BENCH}"
    python tools/eval_bench/build_scannetpp_val8.py --out "${BENCH}" --workers 16 ${N_CLIPS:+--n ${N_CLIPS}}
fi

# 2) official Pi3 (reference; trained on ScanNet++, so the held-out scenes may not be unseen for it)
if selected pi3_official; then
    if [ -f "${PI3_WEIGHTS}" ]; then
        run_infer pi3_official --model-size dinov2_large_0.95B --weights "${PI3_WEIGHTS}"
    else
        echo "[WARN] official Pi3 weights not found: ${PI3_WEIGHTS} (set PI3_WEIGHTS=...), skipped"
    fi
fi

# 3) training runs
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name model run k <<< "${entry}"
    selected "${name}" || continue
    ck="$(resolve_ckpt "${run}" || true)"
    if [ -z "${ck}" ]; then
        echo "[WARN] ${name}: no ${CKPT} checkpoint under ${RUN_ROOT}/${run}/ckpts (run mox_download.sh first), skipped"
        continue
    fi
    echo "    ${name}: ${ck}"
    run_infer "${name}" --model "${model}" --weights "${ck}" ${k:+--k ${k}}
done

# 4) score everything under BENCH
python tools/eval_bench/evaluate.py "${BENCH}" --workers 32
echo "==> ${BENCH}/eval/summary.md"
