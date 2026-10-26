#!/usr/bin/env bash
# One-command evaluation of the SceneGeneration exp1022 runs (and the official Pi3) on the ScanNet++ held-out split
# of training (dataset scannetpp_r3), through the regular infer -> eval -> vis engine of this benchmark.
#
#   bash benchmark/scripts/run_exp1022_scannetpp_r3.sh                  # official Pi3 + exp1022_00..05, latest ckpts
#   SMOKE=1 bash benchmark/scripts/run_exp1022_scannetpp_r3.sh          # 2 scenes only, separate output dir
#   MODELS="exp1022_00 exp1022_03" NPU_ID=1 bash benchmark/scripts/run_exp1022_scannetpp_r3.sh
#   KSWEEP=1 bash ...                                                   # + loop runs at smaller inference K
#   CKPT_NAME=best_model bash ...                                       # checkpoint_N | best_model | latest
#
# Inputs (edit below or override from the environment):
#   DATA_ROOT  ScanNet++ in the reliev3r layout (the training data root; split = last 10% of the scenes)
#   EXP_ROOT   <run>/ckpts/checkpoint_N, as written by run_sh_s3/run_scannetpp/mox_download.sh
#   PI3_CKPT   official Pi3 weights (model.safetensors)
# Results: benchmark/${OUT_ROOT}/scannetpp_r3/test_stride25_maxall_len8_numseq0_<res>/metrics_<model>.xlsx and a
# combined table summary_exp1022.md next to them. Finished sequences / metrics are reused on re-runs.
set -euo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BENCH}"

# -----------------------------
# Paths / resources
# -----------------------------
DATA_ROOT="${DATA_ROOT:-/efs-m3ga/z00533370/reliev3r_data/scannetpp}"
EXP_ROOT="${EXP_ROOT:-/efs-m3ga/z00533370/exp/pi3sup/outputs}"
PI3_CKPT="${PI3_CKPT:-./ckpts/Pi3/model.safetensors}"
CKPT_NAME="${CKPT_NAME:-latest}"
MODELS="${MODELS:-pi3 exp1022_00 exp1022_01 exp1022_02 exp1022_03 exp1022_04 exp1022_05}"
NPU_ID="${NPU_ID:-0}"                # one NPU; run several invocations with different MODELS / NPU_ID to parallelise
DTYPE="${DTYPE:-fp32}"               # as for the other benchmark models
OUT_ROOT="${OUT_ROOT:-outputs}"

# exp1022 runs: name | model spec | loop body pairs | inference K
declare -A RUN SPEC LAYERS KINF
RUN[exp1022_00]=exp1022_00_scannetpp_base_dec384x36_k1_head2;  SPEC[exp1022_00]=dinov3_small_w384;    LAYERS[exp1022_00]=0; KINF[exp1022_00]=0
RUN[exp1022_01]=exp1022_01_scannetpp_loop384_L1_k14to18_head2; SPEC[exp1022_01]=dinov3_small_w384;    LAYERS[exp1022_01]=1; KINF[exp1022_01]=18
RUN[exp1022_02]=exp1022_02_scannetpp_loop384_L2_k5to9_head2;   SPEC[exp1022_02]=dinov3_small_w384;    LAYERS[exp1022_02]=2; KINF[exp1022_02]=9
RUN[exp1022_03]=exp1022_03_scannetpp_loop384_L6_k2to3_head2;   SPEC[exp1022_03]=dinov3_small_w384;    LAYERS[exp1022_03]=6; KINF[exp1022_03]=3
RUN[exp1022_04]=exp1022_04_scannetpp_loop384_L1_k1to18_head2;  SPEC[exp1022_04]=dinov3_small_w384;    LAYERS[exp1022_04]=1; KINF[exp1022_04]=18
RUN[exp1022_05]=exp1022_05_scannetpp_base_dec384x12_k1_head2;  SPEC[exp1022_05]=dinov3_small_w384x12; LAYERS[exp1022_05]=0; KINF[exp1022_05]=0
# extra inference K for KSWEEP=1 (loop runs only)
KSWEEP_LIST="exp1022_01:14 exp1022_01:16 exp1022_02:5 exp1022_02:7 exp1022_03:2 exp1022_04:6 exp1022_04:12"

# -----------------------------
# Checks
# -----------------------------
[ -f scripts/run_infer.py ] && [ -f datasets/scannetpp_r3.py ] && [ -d models/pi3loop ] || {
    echo "[ERROR] ${BENCH} is not benchmark/ of branch xiaoxu/bench_loop (scannetpp_r3 / pi3loop missing)" >&2; exit 1; }
[ -d "${DATA_ROOT}" ] || { echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}" >&2; exit 1; }
python - <<'PY' || { echo "[ERROR] python packages missing for scripts/run_infer.py (see above)" >&2; exit 1; }
import importlib.util, sys
missing = [m for m in ("torch", "torch_npu", "deepspeed", "omegaconf", "safetensors", "transformers", "imageio",
                       "openpyxl", "pandas", "tqdm") if importlib.util.find_spec(m) is None]
if missing:
    print("missing:", " ".join(missing)); sys.exit(1)
PY

SMOKE="${SMOKE:-0}"
if [ "${SMOKE}" = "1" ]; then
    echo "==> SMOKE: 2 held-out scenes only"
fi

ckpt_of () {   # <model key> -> checkpoint dir
    local d="${EXP_ROOT}/${RUN[$1]}/ckpts"
    if [ "${CKPT_NAME}" = "latest" ]; then
        ls -d "${d}"/checkpoint_* 2>/dev/null | sort -V | tail -1
    else
        [ -e "${d}/${CKPT_NAME}" ] && echo "${d}/${CKPT_NAME}"
    fi
}

LOGDIR="${OUT_ROOT}/scannetpp_r3/logs"
mkdir -p "${LOGDIR}"
FAILED=()

run_one () {   # <model key> [K]: one model through engine.sh, in a subshell so variables do not leak
    local key="$1" k="${2:-}" res tag
    tag="${key}${k:+_K${k}}"
    if [ "${key}" = "pi3" ]; then res=518x350; else res=512x336; fi
    echo "================ ${tag} (${res})  $(date '+%F %T') ================"
    (
        set +u
        PYTHON="python"; SCRIPT="scripts/run_infer.py"; EVAL_SCRIPT="scripts/run_eval.py"; VIS_SCRIPT="scripts/run_vis.py"
        source "./scripts/config/scannetpp_r3/${res}.sh"
        [ "${SMOKE}" = "1" ] && NUM_SEQ=2
        if [ "${key}" = "pi3" ]; then
            [ -f "${PI3_CKPT}" ] || { echo "[ERROR] official Pi3 weights not found: ${PI3_CKPT} (set PI3_CKPT)"; exit 3; }
            MODEL="pi3"; MODEL_SIZE="dinov2_large_0.95B"; CONFIG=""; CKPT="${PI3_CKPT}"; PREFIX="pi3_official"
            LOOP_LAYERS=""; LOOP_K=""
        else
            CKPT="$(ckpt_of "${key}" || true)"
            [ -n "${CKPT}" ] || { echo "[ERROR] no ${CKPT_NAME} checkpoint under ${EXP_ROOT}/${RUN[$key]}/ckpts"; exit 3; }
            MODEL="pi3loop"; MODEL_SIZE="${SPEC[$key]}"; CONFIG=""
            LOOP_LAYERS="${LAYERS[$key]}"; LOOP_K="${k:-${KINF[$key]}}"
            PREFIX="${key}$( [ "${CKPT_NAME}" = latest ] || echo "_${CKPT_NAME}" )${k:+_K${k}}"
            echo "[run] ${key}: ${CKPT} (${MODEL_SIZE}, loop_layers=${LOOP_LAYERS}, K=${LOOP_K})"
        fi
        USE_GT_POSE=0; USE_GT_K=0
        DEVICE="npu:0"                          # NPU_ID is exported as ASCEND_RT_VISIBLE_DEVICES by engine.sh
        POSE_THRESHOLD=30; DEPTH_ALIGN="scale_shift"; EVAL_DEVICE="${DEVICE}"
        DEPTH_VMIN=0.1; DEPTH_VMAX=10.0
        REUSE_PRED_IF_EXISTS=1
        source ./scripts/lib/engine.sh
    ) > "${LOGDIR}/${tag}.log" 2>&1 && echo "   ok (log ${LOGDIR}/${tag}.log)" || {
        echo "   FAILED, see ${LOGDIR}/${tag}.log"; tail -5 "${LOGDIR}/${tag}.log" | sed 's/^/   | /'; FAILED+=("${tag}"); }
}

export DATA_ROOT
for m in ${MODELS}; do
    if [ "${m}" != pi3 ] && [ -z "${RUN[$m]:-}" ]; then echo "[WARN] unknown model ${m}, skipped"; continue; fi
    run_one "${m}"
done
if [ "${KSWEEP:-0}" = "1" ]; then
    for e in ${KSWEEP_LIST}; do
        m="${e%%:*}"; k="${e##*:}"
        [[ " ${MODELS} " == *" ${m} "* ]] && run_one "${m}" "${k}"
    done
fi

# -----------------------------
# Combined table of the summary sheets
# -----------------------------
python - "${OUT_ROOT}/scannetpp_r3" <<'PY'
import glob, os, sys
import pandas as pd
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "metrics_*.xlsx"))):
    try:
        s = pd.read_excel(f, sheet_name="summary", header=None)
    except Exception as e:  # noqa: BLE001
        print(f"[summary] skip {f}: {e}"); continue
    r = {"setting": os.path.basename(os.path.dirname(f)), "model": os.path.basename(f)[8:-5]}
    group = ""
    for _, row in s.iterrows():
        v = [x for x in row.tolist() if not (isinstance(x, float) and pd.isna(x))]
        if len(v) >= 2 and isinstance(v[-1], (int, float)):
            name = str(v[-2]).replace("\n", " ")
            if len(v) >= 3 and isinstance(v[-3], str) and v[-3] not in ("scannetpp_r3",):
                group = v[-3]
            r[f"{group} {name}".strip()] = round(float(v[-1]), 4)
    rows.append(r)
if rows:
    df = pd.DataFrame(rows)
    out = os.path.join(root, "summary_exp1022.md")
    try:
        table = df.to_markdown(index=False)
    except ImportError:                      # tabulate not installed
        table = df.to_string(index=False)
    with open(out, "w") as fh:
        fh.write(table + "\n")
    print(df.to_string(index=False))
    print(f"==> {out}")
PY

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[WARN] failed: ${FAILED[*]}"
    exit 1
fi
echo "==> done"
