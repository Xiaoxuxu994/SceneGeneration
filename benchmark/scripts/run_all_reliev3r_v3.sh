#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ---- all dataset ----
DATASETS=(
  7scenes
  nrgbd
  dtu
  # scannet
)

# ----- model ------
MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# CKPT="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260104/exp/Reliev3r_260105_v8b2c32s1_m400_n-1/ckpt/30000.ckpt"
# CONFIG="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260104/exp/Reliev3r_260105_v8b2c32s1_m400_n-1/config.json"

CKPT="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260108/exp/Reliev3R_260108_v8b2c32s1_n-1_rdepth_Ipred/ckpt/15000.ckpt"
CONFIG="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260104/exp/Reliev3r_260105_v8b2c32s1_m400_n-1/config.json"

# ---- params ------
SPLIT="test"
SEQ=""   # e.g. chess/seq-01 ; empty means all

MODE="stride"       # stride | random
STRIDE=10           # default stride
MAX_SEQ_LEN=8       # default max seq len
SEED=0              # used when MODE=random

DEVICE="npu:7"
DTYPE="fp32"

OUT_ROOT="/mnt/nvme3/z00533370/3drecon/benchmark/"
PREFIX="260108_v8b2c32s1_n-1_rdepth_Ipred_ckpt15k_v3"           # e.g., "0107_ckpt15000" ; empty -> no suffix

# ---- eval common params ----
POSE_THRESHOLD=30
DEPTH_ALIGN="scale_shift"
EVAL_DEVICE="${DEVICE}"

# ---- vis params ----
DEPTH_VMIN=0.1
DEPTH_VMAX=10.0

# ---- behavior ----
REUSE_PRED_IF_EXISTS=1   # 1: if pred exists, skip infer; 0: always run infer

# -------------------------
# Pretty printing helpers
# -------------------------
banner_dataset () {
  local idx="$1"
  local total="$2"
  local dataset="$3"
  echo
  echo "##########################################################################################"
  echo "###########################  [${idx}/${total}] DATASET: ${dataset}  ###########################"
  echo "###########################  TIME: $(date '+%F %T')  ########################################"
  echo "##########################################################################################"
  echo
}

banner_stage () {
  local stage="$1"
  echo
  echo "=========================================================================================="
  echo "================================= [STAGE] ${stage} ======================================="
  echo "=========================================================================================="
}

banner_done () {
  local idx="$1"
  local total="$2"
  local dataset="$3"
  echo
  echo "##########################################################################################"
  echo "############################  [${idx}/${total}] DONE: ${dataset}  ############################"
  echo "############################  TIME: $(date '+%F %T')  ########################################"
  echo "##########################################################################################"
  echo
}

# -------------------------
# pred helper
# "ready" if at least one pred_extrinsics.npy exists under pred dir
# -------------------------
pred_ready () {
  local d="$1"
  [[ -d "${d}" ]] || return 1
  [[ -n "$(find "${d}" -type f -name "pred_extrinsics.npy" -print -quit 2>/dev/null)" ]]
}
# -------------------------
# VIS helper
# vis output expected at "${DIR}_vis"
# e.g. DIR=.../gt -> .../gt_vis
# -------------------------
run_vis_if_needed () {
  local DIR="$1"
  local VIS_DIR="${DIR}_vis"

  if [[ -d "${VIS_DIR}" ]] && [[ -n "$(ls -A "${VIS_DIR}" 2>/dev/null)" ]]; then
    echo "[VIS] skip (exists & non-empty): ${VIS_DIR}"
    return 0
  fi

  echo "[VIS] run: dir=${DIR} (vmin=${DEPTH_VMIN}, vmax=${DEPTH_VMAX})"
  ${PYTHON} ${VIS_SCRIPT} \
    --dir "${DIR}" \
    --depth_vmin "${DEPTH_VMIN}" \
    --depth_vmax "${DEPTH_VMAX}"
}

TOTAL=${#DATASETS[@]}
IDX=0

for DATASET in "${DATASETS[@]}"; do
  IDX=$((IDX+1))
  banner_dataset "${IDX}" "${TOTAL}" "${DATASET}"

  # -------------------------
  # per-dataset override (STRIDE / MAX_SEQ_LEN / optional MODE)
  # edit these to your desired values
  # -------------------------
  MODE_CUR="${MODE}"
  STRIDE_CUR="${STRIDE}"
  MAX_SEQ_LEN_CUR="${MAX_SEQ_LEN}"

  case "${DATASET}" in
    7scenes)
      STRIDE_CUR=1
      MAX_SEQ_LEN_CUR=8
      ;;
    nrgbd)
      STRIDE_CUR=1
      MAX_SEQ_LEN_CUR=8
      ;;
    dtu)
      STRIDE_CUR=1
      MAX_SEQ_LEN_CUR=8
      ;;
    scannet)
      STRIDE_CUR=1
      MAX_SEQ_LEN_CUR=8
      ;;
    *)
      ;;
  esac

  echo "[RUNTIME] device=${DEVICE} dtype=${DTYPE} eval_device=${EVAL_DEVICE}"
  echo "[CONFIG ] model=${MODEL} prefix=${PREFIX}"
  echo "[CONFIG ] ckpt=${CKPT}"
  echo "[CONFIG ] config=${CONFIG}"
  echo "[CONFIG ] OUT_ROOT=${OUT_ROOT}"
  echo "[CONFIG ] split=${SPLIT} mode=${MODE_CUR} stride=${STRIDE_CUR} max_seq_len=${MAX_SEQ_LEN_CUR} seed=${SEED}"
  [[ -n "${SEQ}" ]] && echo "[CONFIG ] seq=${SEQ}"

  # -------------------------
  # compute ROOT_DIR / PRED_DIR early (for reuse pred)
  # -------------------------
  if [ "${MAX_SEQ_LEN_CUR}" -gt 0 ] 2>/dev/null; then
    MSL="${MAX_SEQ_LEN_CUR}"
  else
    MSL="all"
  fi

  # pred_subdir = model_prefix
  if [[ -n "${PREFIX}" ]]; then
    PRED_SUBDIR="${MODEL}_${PREFIX}"
  else
    PRED_SUBDIR="${MODEL}"
  fi

  if [ "${MODE_CUR}" = "stride" ]; then
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_stride${STRIDE_CUR}_max${MSL}"
  else
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_rand${MSL}_seed${SEED}"
  fi

  GT_DIR="${ROOT_DIR}/gt"
  PRED_DIR="${ROOT_DIR}/${PRED_SUBDIR}"

  # -------------------------
  # 0) INFER
  # -------------------------
  banner_stage "INFER"
  t0=$(date +%s)

  if [[ "${REUSE_PRED_IF_EXISTS}" -eq 1 ]] && pred_ready "${PRED_DIR}"; then
    echo "[INFER] skip: found existing predictions under ${PRED_DIR}"

  else
    EXTRA_ARGS=()
    [[ -n "${SEQ}" ]]    && EXTRA_ARGS+=(--seq "${SEQ}")
    [[ -n "${CKPT}" ]]   && EXTRA_ARGS+=(--ckpt "${CKPT}")
    [[ -n "${CONFIG}" ]] && EXTRA_ARGS+=(--config "${CONFIG}")
    [[ -n "${PREFIX}" ]] && EXTRA_ARGS+=(--prefix "${PREFIX}")

    echo "[INFER] dataset=${DATASET} split=${SPLIT} mode=${MODE_CUR} stride=${STRIDE_CUR} max_seq_len=${MAX_SEQ_LEN_CUR}"
    ${PYTHON} ${SCRIPT} \
      --dataset "${DATASET}" \
      --split "${SPLIT}" \
      --mode "${MODE_CUR}" \
      --stride "${STRIDE_CUR}" \
      --max_seq_len "${MAX_SEQ_LEN_CUR}" \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --out_root "${OUT_ROOT}" \
      "${EXTRA_ARGS[@]}"
  fi
  t1=$(date +%s)
  echo "[TIME] INFER seconds=$((t1-t0))"

  # -------------------------
  # 1) EVAL
  # -------------------------
  banner_stage "EVAL"
  t0=$(date +%s)

  # per-dataset POINTMAP_THRESHOLD
  case "${DATASET}" in
    7scenes) POINTMAP_THRESHOLD=0.1 ;;
    nrgbd)   POINTMAP_THRESHOLD=0.1 ;;
    dtu)     POINTMAP_THRESHOLD=0.1 ;;
    scannet) POINTMAP_THRESHOLD=0.1 ;;
    *)       POINTMAP_THRESHOLD=0.1 ;;
  esac

  # per-model PREDICTED_POINTMAP
  case "${MODEL}" in
    vggt)     PREDICTED_POINTMAP=1 ;;
    pi3)      PREDICTED_POINTMAP=1 ;;
    reliev3r) PREDICTED_POINTMAP=0 ;;
    *)        PREDICTED_POINTMAP=0 ;;
  esac

  echo "[EVAL] root=${ROOT_DIR}"
  echo "[EVAL] gt_subdir=gt  pred_subdir=${PRED_SUBDIR}"
  echo "[EVAL] pointmap_threshold=${POINTMAP_THRESHOLD}  predicted_pointmap=${PREDICTED_POINTMAP}"
  echo "[PATH] GT_DIR=${GT_DIR}"
  echo "[PATH] PRED_DIR=${PRED_DIR}"

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

  t1=$(date +%s)
  echo "[TIME] EVAL seconds=$((t1-t0))"

  # -------------------------
  # 2) VIS
  # -------------------------
  banner_stage "VIS"
  t0=$(date +%s)

  echo "[VIS] gt_dir=${GT_DIR}  (expect vis: ${GT_DIR}_vis)"
  echo "[VIS] pred_dir=${PRED_DIR} (expect vis: ${PRED_DIR}_vis)"

  # requirement: if gt has no gt_vis -> run; if exists -> skip
  run_vis_if_needed "${GT_DIR}"
  run_vis_if_needed "${PRED_DIR}"

  t1=$(date +%s)
  echo "[TIME] VIS seconds=$((t1-t0))"

  banner_done "${IDX}" "${TOTAL}" "${DATASET}"
done
