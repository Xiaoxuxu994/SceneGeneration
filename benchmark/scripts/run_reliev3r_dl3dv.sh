#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ----- model ------
MODEL="reliev3r" # ['vggt','pi3','reliev3r']
MODEL_SIZE=""
source ./scripts/config/dl3dv/476x266.sh
# source ./scripts/config/dl3dv/518x294.sh

# CKPT="./ckpts/Pi3/model.safetensors"
# CONFIG=""
# PREFIX="Reliev3R-260415-mixSupervise-lr1e-5-All-SVallSynthReal-Pi3-freeze-newcode_ckptpi3"

CKPT="experiments/Reliev3R_bs2v8_260424_DL3DV_ss_aug_rgst/ckpt/20000.ckpt"
CONFIG="experiments/Reliev3R_bs2v8_260424_DL3DV_ss_aug_rgst/config.json"
PREFIX="Reliev3R_bs2v8_260424_DL3DV_ss_aug_rgst_ckpt20k"

# CKPT="experiments/Reliev3R_bs2v8_260421_SpvidDL3DV/ckpt/70000.ckpt"
# CONFIG="experiments/Reliev3R_bs2v8_260421_SpvidDL3DV/config.json"
# PREFIX="Reliev3R_bs2v8_260421_SpvidDL3DV_ckpt70k"

# MODEL="pi3" # ['vggt','pi3','reliev3r']

# MODEL_SIZE="dinov3_huge+_3Bv2"
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0410_pi3_3B_datamix_sync_v2_highres_v2-16-m16_512x336_lr1e4/ckpts/model_ckpt19"
# PREFIX="exp0410_pi3_3B_datamix_sync_v2_highres_v2-16-m16_512x336_lr1e4_ckpt19"
# CONFIG=""
# source ./scripts/config/ase_512x336.sh

# MODEL_SIZE="dinov2_large_0.95B"
# CKPT="./ckpts/Pi3/model.safetensors"
# PREFIX="pi3_official"
# CONFIG=""
# source ./scripts/config/ase_518x336.sh

# MODEL="vggt" # ['vggt','pi3','reliev3r']

# MODEL_SIZE=""
# CONFIG="./ckpts/VGGT-1B/config.json"
# CKPT="./ckpts/VGGT-1B/model.pt"
# PREFIX="vggt_official"
# source ./scripts/config/ase/518x336.sh

# CONFIG=""

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch023-loss_valid0.2258-metric_valid0.1446.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep23"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch015-loss_valid0.2459-metric_valid0.1737.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep15"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch013-loss_valid0.2557-metric_valid0.1859.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep13"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch011-loss_valid0.2917-metric_valid0.2021.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep11"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch009-loss_valid0.2741-metric_valid0.2231.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep9"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch007-loss_valid0.3137-metric_valid0.2497.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep7"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch005-loss_valid0.3839-metric_valid0.2823.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep5"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch003-loss_valid0.4962-metric_valid0.3424.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep3"

# CKPT="/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch001-loss_valid0.5604-metric_valid0.4377.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_ep1"


USE_GT_POSE=0 # 1 if using GT extrinsics; 0 predicted
USE_GT_K=0 # 1 if using GT intrinsics; 0 predicted

SKIP_POINTMAP=0 # 1 if skip pointmap evaluation
SKIP_NORMAL=1 # 1 if skip point normal evaluation

NPU_ID="1"
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

  echo "[RUNTIME] device=${DEVICE} dtype=${DTYPE} eval_device=${EVAL_DEVICE}"
  echo "[CONFIG ] model=${MODEL} prefix=${PREFIX}"
  echo "[CONFIG ] ckpt=${CKPT}"
  echo "[CONFIG ] config=${CONFIG}"
  echo "[CONFIG ] OUT_ROOT=${OUT_ROOT}"
  echo "[CONFIG ] split=${SPLIT} mode=${MODE} stride=${STRIDE} seq_len=${SEQ_LEN} max_seq_len=${MAX_SEQ_LEN} num_seq=${NUM_SEQ} seed=${SEED}"
  [[ -n "${SEQ}" ]] && echo "[CONFIG ] seq=${SEQ}"

  # -------------------------
  # compute ROOT_DIR / PRED_DIR early (for reuse pred)
  # -------------------------
  if [ "${MAX_SEQ_LEN}" -gt 0 ] 2>/dev/null; then
    MSL="${MAX_SEQ_LEN}"
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

  if [ "${MODE}" = "stride" ]; then
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_stride${STRIDE}_max${MSL}_len${SEQ_LEN}_numseq${NUM_SEQ}_${RGB_RESIZE_W}x${RGB_RESIZE_H}"
  else
    ROOT_DIR="${OUT_ROOT}/${DATASET}/${SPLIT}_rand_max${MSL}_len${SEQ_LEN}_numseq${NUM_SEQ}_${RGB_RESIZE_W}x${RGB_RESIZE_H}"
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

    echo "[INFER] dataset=${DATASET} split=${SPLIT} mode=${MODE}"
    echo "[INFER] stride=${STRIDE} max_seq_len=${MSL} seq_len=${SEQ_LEN} num_seq=${NUM_SEQ}"
    echo "[INFER] use_gt_pose=${USE_GT_POSE} use_gt_k=${USE_GT_K}"
    ASCEND_RT_VISIBLE_DEVICES=${NPU_ID} ${PYTHON} ${SCRIPT} \
      --dataset "${DATASET}" \
      --split "${SPLIT}" \
      --mode "${MODE}" \
      --stride "${STRIDE}" \
      --seq_len "${SEQ_LEN}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --num_seq "${NUM_SEQ}" \
      --model "${MODEL}" \
      --model_size "${MODEL_SIZE}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --out_root "${OUT_ROOT}" \
      --use_gt_pose "${USE_GT_POSE}" \
      --use_gt_k "${USE_GT_K}" \
      --rgb_origin_W "${RGB_ORIGIN_W}" \
      --rgb_origin_H "${RGB_ORIGIN_H}" \
      --crop_left "${CROP_LEFT}" \
      --crop_right "${CROP_RIGHT}" \
      --crop_top "${CROP_TOP}" \
      --crop_bottom "${CROP_BOTTOM}" \
      --rgb_resize_W "${RGB_RESIZE_W}" \
      --rgb_resize_H "${RGB_RESIZE_H}" \
      --depth_scale "${DEPTH_SCALE}" \
      --depth_min "${DEPTH_MIN}" \
      --depth_max "${DEPTH_MAX}" \
      --fx "${FX}" \
      --fy "${FY}" \
      --cx "${CX}" \
      --cy "${CY}" \
      "${EXTRA_ARGS[@]}"
  fi
  t1=$(date +%s)
  echo "[TIME] INFER seconds=$((t1-t0))"

  # # -------------------------
  # # 1) EVAL
  # # -------------------------
  # banner_stage "EVAL"
  # t0=$(date +%s)

  # # per-dataset POINTMAP_THRESHOLD
  # case "${DATASET}" in
  #   7scenes) POINTMAP_THRESHOLD=0.1 ;;
  #   nrgbd)   POINTMAP_THRESHOLD=0.1 ;;
  #   dtu)     POINTMAP_THRESHOLD=0.1 ;;
  #   scannet) POINTMAP_THRESHOLD=0.1 ;;
  #   *)       POINTMAP_THRESHOLD=0.1 ;;
  # esac

  # # per-model PREDICTED_POINTMAP
  # case "${MODEL}" in
  #   vggt)     PREDICTED_POINTMAP=0 ;;
  #   pi3)      PREDICTED_POINTMAP=1 ;;
  #   reliev3r) PREDICTED_POINTMAP=0 ;;
  #   *)        PREDICTED_POINTMAP=0 ;;
  # esac

  # echo "[EVAL] root=${ROOT_DIR}"
  # echo "[EVAL] gt_subdir=gt  pred_subdir=${PRED_SUBDIR}"
  # echo "[EVAL] pointmap_threshold=${POINTMAP_THRESHOLD}  predicted_pointmap=${PREDICTED_POINTMAP}"
  # echo "[EVAL] skip_pointmap=${SKIP_POINTMAP}  skip_normal=${SKIP_NORMAL}"
  # echo "[PATH] GT_DIR=${GT_DIR}"
  # echo "[PATH] PRED_DIR=${PRED_DIR}"

  # ASCEND_RT_VISIBLE_DEVICES=${NPU_ID} ${PYTHON} ${EVAL_SCRIPT} \
  #   --root "${ROOT_DIR}" \
  #   --dataset_name "${DATASET}" \
  #   --gt_subdir "gt" \
  #   --pred_subdir "${PRED_SUBDIR}" \
  #   --pose_threshold "${POSE_THRESHOLD}" \
  #   --pointmap_threshold "${POINTMAP_THRESHOLD}" \
  #   --predicted_pointmap "${PREDICTED_POINTMAP}" \
  #   --skip_pointmap "${SKIP_POINTMAP}" \
  #   --skip_normal "${SKIP_NORMAL}" \
  #   --depth_align "${DEPTH_ALIGN}" \
  #   --device "${EVAL_DEVICE}"

  # t1=$(date +%s)
  # echo "[TIME] EVAL seconds=$((t1-t0))"

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
