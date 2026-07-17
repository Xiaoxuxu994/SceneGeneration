#!/usr/bin/env bash
set -e

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py" 

# ---- all datset----
DATASETS=(
  7scenes
  nrgbd
  # dtu
  # scannet
)

# ----- model ------
# MODEL="pi3"
# CKPT=""
# CONFIG=""

MODEL="reliev3r"
# ### 20250106
# CKPT="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260104/exp/Reliev3r_260105_v8b2c32s1_m400_n-1/ckpt/5000.ckpt"
# CONFIG="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260104/exp/Reliev3r_260105_v8b2c32s1_m400_n-1/config.json"
### 20250107
CKPT="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260104/exp/Reliev3r_260105_v8b2c32s1_m400_n-1/ckpt/15000.ckpt"
CONFIG="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260104/exp/Reliev3r_260105_v8b2c32s1_m400_n-1/config.json"

# ---- param ------
SPLIT="test"
SEQ=""   # chess/seq-01

MODE="stride"       # stride | random
STRIDE=10
MAX_SEQ_LEN=8
DEVICE="npu:5"
DTYPE="fp32"

OUT_ROOT="/mnt/nvme3/z00533370/3drecon/benchmark/debug/"
PREFIX=""           # e.g., "0107_ckpt15000"

# ---- eval common params ----
POSE_THRESHOLD=30
DEPTH_ALIGN="scale_shift"
EVAL_DEVICE=$DEVICE 

TOTAL=${#DATASETS[@]}
IDX=0

for DATASET in "${DATASETS[@]}"; do

  echo
  echo "##########################################################################################"
  echo "###############################  [${IDX}/${TOTAL}] DATASET: ${DATASET}  ###############################"
  echo "###############################  TIME: $(date '+%F %T')  ############################################"
  echo "##########################################################################################"
  echo

  # 0.0 Run Inference
  echo "============================== [STAGE] INFER =============================="
  echo "=========================================================================================="
  echo "[INFER] dataset=${DATASET} split=${SPLIT} mode=${MODE} stride=${STRIDE} max_seq_len=${MAX_SEQ_LEN}"
  echo "[INFER] model=${MODEL} prefix=${PREFIX}"
  echo "[INFER] ckpt=${CKPT}"
  echo "[INFER] config=${CONFIG}"
  echo "[INFER] OUT_ROOT=${OUT_ROOT}"
  echo "=========================================================================================="

  EXTRA_ARGS=()
  [[ -n "${SEQ}" ]]    && EXTRA_ARGS+=(--seq "${SEQ}")
  [[ -n "${CKPT}" ]]   && EXTRA_ARGS+=(--ckpt "${CKPT}")
  [[ -n "${CONFIG}" ]] && EXTRA_ARGS+=(--config "${CONFIG}")
  [[ -n "${PREFIX}" ]] && EXTRA_ARGS+=(--prefix "${PREFIX}")

  ${PYTHON} ${SCRIPT} \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --mode "${MODE}" \
    --stride "${STRIDE}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --out_root "${OUT_ROOT}" \
    "${EXTRA_ARGS[@]}"

  # 1.0 Run Eval
  echo "============================== [STAGE] EVAL =============================="
  # ---- pred_subdir = model_prefix ----
  if [[ -n "${PREFIX}" ]]; then
    PRED_SUBDIR="${MODEL}_${PREFIX}"
  else
    PRED_SUBDIR="${MODEL}"
  fi

  # ---- per-dataset POINTMAP_THRESHOLD ----
  case "${DATASET}" in
    7scenes) POINTMAP_THRESHOLD=0.1 ;;
    nrgbd)   POINTMAP_THRESHOLD=0.1 ;;
    dtu)     POINTMAP_THRESHOLD=100 ;; 
    scannet) POINTMAP_THRESHOLD=0.1 ;;    
    *)       POINTMAP_THRESHOLD=0.1 ;;
  esac

  # ---- per-model PREDICTED_POINTMAP ----
  case "${MODEL}" in
    vggt)     PREDICTED_POINTMAP=1 ;;
    pi3)      PREDICTED_POINTMAP=1 ;;
    reliev3r) PREDICTED_POINTMAP=0 ;;
    *)        PREDICTED_POINTMAP=0 ;;
  esac

  if [ "${MAX_SEQ_LEN}" -gt 0 ] 2>/dev/null; then
    MSL="${MAX_SEQ_LEN}"
  else
    MSL="all"
  fi

  if [ "${MODE}" = "stride" ]; then
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_stride${STRIDE}_max${MSL}"
  else
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_rand${MSL}_seed${SEED}"
  fi

  echo "------------------------------------------------------------------------------------------"
  echo "[EVAL] root=${ROOT_DIR}"
  echo "[EVAL] gt_subdir=gt  pred_subdir=${PRED_SUBDIR}"
  echo "[EVAL] pointmap_threshold=${POINTMAP_THRESHOLD}  predicted_pointmap=${PREDICTED_POINTMAP}"
  echo "------------------------------------------------------------------------------------------"

  ${PYTHON} ${EVAL_SCRIPT} \
    --root "${ROOT_DIR}" \
    --dataset_name "${DATASET}" \
    --gt_subdir "gt" \
    --pred_subdir "${PRED_SUBDIR}" \
    --pose_threshold "${POSE_THRESHOLD}" \
    --pointmap_threshold "${POINTMAP_THRESHOLD}" \
    --predicted_pointmap "${PREDICTED_POINTMAP}" \
    --depth_align "${DEPTH_ALIGN}" \
    --device "${EVAL_DEVICE}"
  
  # 2.0 Run Visualization
  DEPTH_VMIN=0.1
  DEPTH_VMAX=10.0

  # -------------------------
  # VIS (per-iteration)
  # vis 输出到 "${DIR}_vis", DIR=.../gt -> .../gt_vis
  # -------------------------
  run_vis_if_needed () {
    local DIR="$1"
    local VIS_DIR="${DIR}_vis"

    # 如果 vis 目录存在且非空，就跳过
    if [[ -d "${VIS_DIR}" ]] && [[ -n "$(ls -A "${VIS_DIR}" 2>/dev/null)" ]]; then
      echo "[VIS] skip (exists): ${VIS_DIR}"
      return 0
    fi

    echo "[VIS] run: dir=${DIR}  (vmin=${DEPTH_VMIN}, vmax=${DEPTH_VMAX})"
    ${PYTHON} ${VIS_SCRIPT} \
      --dir "${DIR}" \
      --depth_vmin "${DEPTH_VMIN}" \
      --depth_vmax "${DEPTH_VMAX}"
  }

  GT_DIR="${ROOT_DIR}/gt"
  PRED_DIR="${ROOT_DIR}/${PRED_SUBDIR}"

  run_vis_if_needed "${GT_DIR}"
  run_vis_if_needed "${PRED_DIR}"

  echo
  echo "##########################################################################################"
  echo "###############################  [${IDX}/${TOTAL}] DONE: ${DATASET}  #################################"
  echo "###############################  TIME: $(date '+%F %T')  ############################################"
  echo "##########################################################################################"
  echo
  IDX=$((IDX+1))
done