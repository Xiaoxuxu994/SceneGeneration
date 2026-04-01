#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ----- model ------
MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# CKPT="experiments/Reliev3R_bs2v8_260302_ase10p_fullSup/ckpt/70000.ckpt"
# CONFIG="experiments/Reliev3R_bs2v8_260302_ase10p_fullSup/config.json"
# PREFIX="Reliev3R_bs2v8_260302_ase10p_fullSup_ckpt70k"

CKPT="experiments/Reliev3R-260327-mixSupervise-lr1e-5-All-mixaseBmvsHsimMvssyn-dl3dvRe10k-addTrans0p5Dist0p1Flow5-Pi3/ckpt/35000.ckpt"
CONFIG="experiments/Reliev3R-260327-mixSupervise-lr1e-5-All-mixaseBmvsHsimMvssyn-dl3dvRe10k-addTrans0p5Dist0p1Flow5-Pi3/config.json"
PREFIX="Reliev3R-260327-mixSupervise-lr1e-5-All-mixaseBmvsHsimMvssyn-dl3dvRe10k-addTrans0p5Dist0p1Flow5-Pi3_ckpt35k"

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# CKPT="./ckpt/Pi3/"
# PREFIX="pi3_official"
# CONFIG=""

# MODEL="vggt" # ['vggt','pi3','reliev3r']

# CONFIG="./ckpt/VGGT-1B/config.json"
# CKPT="./ckpt/VGGT-1B/model.pt"
# PREFIX="vggt_official"

# CONFIG=""

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch023-loss_valid0.2258-metric_valid0.1446.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep23"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch007-loss_valid0.3137-metric_valid0.2497.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep7"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch005-loss_valid0.3839-metric_valid0.2823.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep5"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch003-loss_valid0.4962-metric_valid0.3424.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep3"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch001-loss_valid0.5604-metric_valid0.4377.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep1"

# ---- params ------
DATASETS=(
  # 7scenes
  # nrgbd
  # dtu
  scannet
  # dl3dv
  # ase
)

# SPLIT="test"
# SEQ=""   # e.g. chess/seq-01 ; empty means all

# MODE="random"       # stride | random
# STRIDE=5            # default stride
# SEQ_LEN=8           # use N frames in sequence
# MAX_SEQ_LEN=32      # default max seq len, default -1 to use all
# SEED=42             # used when MODE=random
# NUM_SEQ=10

USE_GT_POSE=0 # 1 if using GT extrinsics; 0 predicted
USE_GT_K=0 # 1 if using GT intrinsics; 0 predicted

SKIP_POINTMAP=0 # 1 if skip pointmap evaluation
SKIP_NORMAL=1 # 1 if skip point normal evaluation

NPU_ID="4"
DEVICE="npu:0"
DTYPE="fp32"

OUT_ROOT="outputs"  # "/mnt/nvme3/z00533370/3drecon/benchmark/"

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
    --depth_vmax "${DEPTH_VMAX}" \
    # --scene_list "1512246 1512275 1512279 1512288 1512262 1512277 1512245 1512247 1512272"
}

TOTAL=${#DATASETS[@]}
IDX=0

for DATASET in "${DATASETS[@]}"; do
  IDX=$((IDX+1))
  banner_dataset "${IDX}" "${TOTAL}" "${DATASET}"

  # # -------------------------
  # # per-dataset override (STRIDE / MAX_SEQ_LEN / optional MODE)
  # # edit these to your desired values
  # # -------------------------
  # MODE_CUR="${MODE}"
  # STRIDE_CUR="${STRIDE}"
  # SEQ_LEN_CUR="${SEQ_LEN}"
  # MAX_SEQ_LEN_CUR="${MAX_SEQ_LEN}"

  # case "${DATASET}" in
  #   7scenes)
  #     STRIDE_CUR=40
  #     MAX_SEQ_LEN_CUR=8
  #     ;;
  #   nrgbd)
  #     STRIDE_CUR=100
  #     MAX_SEQ_LEN_CUR=8
  #     ;;
  #   dtu)
  #     STRIDE_CUR=1
  #     MAX_SEQ_LEN_CUR=8
  #     ;;
  #   scannet)
  #     STRIDE_CUR=25
  #     MAX_SEQ_LEN_CUR=8
  #     ;;
  #   *)
  #     ;;
  # esac

  # -------------------------
  # per-dataset override (STRIDE / MAX_SEQ_LEN / optional MODE)
  # edit these to your desired values
  # -------------------------

  case "${DATASET}" in
    7scenes)
      SPLIT="test"
      SEQ=""   # e.g. chess/seq-01 ; empty means all

      MODE_CUR="stride"   # stride | random
      STRIDE_CUR=40       # default stride
      SEQ_LEN_CUR=8       # use N frames in sequence
      MAX_SEQ_LEN_CUR=-1  # default max seq len, default -1 to use all
      SEED=42             # used when MODE=random
      NUM_SEQ=10
      ;;
    ase)
      SPLIT="test"
      SEQ=""   # e.g. chess/seq-01 ; empty means all

      MODE_CUR="random"   # stride | random
      STRIDE_CUR=5        # default stride
      SEQ_LEN_CUR=8       # use N frames in sequence
      MAX_SEQ_LEN_CUR=32  # default max seq len, default -1 to use all
      SEED=42             # used when MODE=random
      NUM_SEQ=10
      ;;
    nrgbd)
      SPLIT="test"
      SEQ=""   # e.g. chess/seq-01 ; empty means all

      MODE_CUR="stride"   # stride | random
      STRIDE_CUR=100      # default stride
      SEQ_LEN_CUR=8       # use N frames in sequence
      MAX_SEQ_LEN_CUR=-1  # default max seq len, default -1 to use all
      SEED=42             # used when MODE=random
      NUM_SEQ=10
      ;;
    dl3dv)
      SPLIT="test"
      SEQ=""   # e.g. chess/seq-01 ; empty means all

      MODE_CUR="stride"   # stride | random
      STRIDE_CUR=5        # default stride
      SEQ_LEN_CUR=8       # use N frames in sequence
      MAX_SEQ_LEN_CUR=-1  # default max seq len, default -1 to use all
      SEED=42             # used when MODE=random
      NUM_SEQ=100
      ;;
    scannet)
      SPLIT="test"
      SEQ=""   # e.g. chess/seq-01 ; empty means all

      MODE_CUR="stride"   # stride | random
      STRIDE_CUR=25       # default stride
      SEQ_LEN_CUR=8       # use N frames in sequence
      MAX_SEQ_LEN_CUR=-1  # default max seq len, default -1 to use all
      SEED=42             # used when MODE=random
      NUM_SEQ=10
      ;;
    *)
      ;;
  esac

  echo "[RUNTIME] device=${DEVICE} dtype=${DTYPE} eval_device=${EVAL_DEVICE}"
  echo "[CONFIG ] model=${MODEL} prefix=${PREFIX}"
  echo "[CONFIG ] ckpt=${CKPT}"
  echo "[CONFIG ] config=${CONFIG}"
  echo "[CONFIG ] OUT_ROOT=${OUT_ROOT}"
  echo "[CONFIG ] split=${SPLIT} mode=${MODE_CUR} stride=${STRIDE_CUR} seq_len=${SEQ_LEN_CUR} max_seq_len=${MAX_SEQ_LEN_CUR} num_seq=${NUM_SEQ} seed=${SEED}"
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

  if [ "${USE_GT_POSE}" -eq 1 ]; then
    PRED_SUBDIR="${PRED_SUBDIR}_gt_pose"
  fi

  if [ "${USE_GT_K}" -eq 1 ]; then
    PRED_SUBDIR="${PRED_SUBDIR}_gt_K"
  fi

  if [ "${MODE_CUR}" = "stride" ]; then
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_stride${STRIDE_CUR}_max${MSL}_len${SEQ_LEN_CUR}_numseq${NUM_SEQ}"
  else
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_rand_max${MSL}_len${SEQ_LEN_CUR}_numseq${NUM_SEQ}"
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

    echo "[INFER] dataset=${DATASET} split=${SPLIT} mode=${MODE_CUR}"
    echo "[INFER] stride=${STRIDE_CUR} max_seq_len=${MSL} seq_len=${SEQ_LEN_CUR} num_seq=${NUM_SEQ}"
    echo "[INFER] use_gt_pose=${USE_GT_POSE} use_gt_k=${USE_GT_K}"
    ASCEND_RT_VISIBLE_DEVICES=${NPU_ID} ${PYTHON} ${SCRIPT} \
      --dataset "${DATASET}" \
      --split "${SPLIT}" \
      --mode "${MODE_CUR}" \
      --stride "${STRIDE_CUR}" \
      --seq_len "${SEQ_LEN_CUR}" \
      --max_seq_len "${MAX_SEQ_LEN_CUR}" \
      --num_seq "${NUM_SEQ}" \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --out_root "${OUT_ROOT}" \
      --use_gt_pose "${USE_GT_POSE}" \
      --use_gt_k "${USE_GT_K}" \
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
    vggt)     PREDICTED_POINTMAP=0 ;;
    pi3)      PREDICTED_POINTMAP=1 ;;
    reliev3r) PREDICTED_POINTMAP=0 ;;
    *)        PREDICTED_POINTMAP=0 ;;
  esac

  echo "[EVAL] root=${ROOT_DIR}"
  echo "[EVAL] gt_subdir=gt  pred_subdir=${PRED_SUBDIR}"
  echo "[EVAL] pointmap_threshold=${POINTMAP_THRESHOLD}  predicted_pointmap=${PREDICTED_POINTMAP}"
  echo "[EVAL] skip_pointmap=${SKIP_POINTMAP}  skip_normal=${SKIP_NORMAL}"
  echo "[PATH] GT_DIR=${GT_DIR}"
  echo "[PATH] PRED_DIR=${PRED_DIR}"

  ASCEND_RT_VISIBLE_DEVICES=${NPU_ID} ${PYTHON} ${EVAL_SCRIPT} \
    --root "${ROOT_DIR}" \
    --dataset_name "${DATASET}" \
    --gt_subdir "gt" \
    --pred_subdir "${PRED_SUBDIR}" \
    --pose_threshold "${POSE_THRESHOLD}" \
    --pointmap_threshold "${POINTMAP_THRESHOLD}" \
    --predicted_pointmap "${PREDICTED_POINTMAP}" \
    --skip_pointmap "${SKIP_POINTMAP}" \
    --skip_normal "${SKIP_NORMAL}" \
    --depth_align "${DEPTH_ALIGN}" \
    --device "${EVAL_DEVICE}"

  t1=$(date +%s)
  echo "[TIME] EVAL seconds=$((t1-t0))"

  # # -------------------------
  # # 2) VIS
  # # -------------------------
  # banner_stage "VIS"
  # t0=$(date +%s)

  # echo "[VIS] gt_dir=${GT_DIR}  (expect vis: ${GT_DIR}_vis)"
  # echo "[VIS] pred_dir=${PRED_DIR} (expect vis: ${PRED_DIR}_vis)"

  # # requirement: if gt has no gt_vis -> run; if exists -> skip
  # run_vis_if_needed "${GT_DIR}"
  # run_vis_if_needed "${PRED_DIR}"

  # t1=$(date +%s)
  # echo "[TIME] VIS seconds=$((t1-t0))"

  banner_done "${IDX}" "${TOTAL}" "${DATASET}"
done
