#!/bin/bash

# Configuration
CKPT_ROOT="<YOUR_CKPT_PATH>"
DATASET_PATH="<YOUR_RAW_DATAPATH>/splits/hq_lvsm_val.txt"
OUT_ROOT="<YOUR_OUT_ROOT>"
RENDER_VIDEO=true

# List of experiment names
EXP_NAMES=(
  "mapa_encoder_ftDecTok_i8"
  "dual_encoder_ft_i8"
  "dual_encoder_ftDecTok_i32"
)

# Default list of input view counts
V_INPUTS=(8 32)
V_TARGETS=(8 32)

# Difficulty levels
DIFFICULTIES=("ae" "easy" "medium" "hard")
EVAL_IDX_ROOT="data/scannetpp/eval_idx"


for EXP_NAME in "${EXP_NAMES[@]}"; do
  CONFIG="configs/${EXP_NAME}.yaml"
  PATCH_SIZE=8

  for V_INPUT in "${V_INPUTS[@]}"; do
    for V_TARGET in "${V_TARGETS[@]}"; do
      for DIFFICULTY in "${DIFFICULTIES[@]}"; do
        V_TOTAL=$((V_TARGET + V_INPUT))

        # Choose VIEW_IDX_FILE adaptively
        VIEW_IDX_FILE="${EVAL_IDX_ROOT}/i${V_INPUT}_t${V_TARGET}_${DIFFICULTY}.json"

        if ! [ -f "$VIEW_IDX_FILE" ]; then
          continue
        fi

        echo "Running inference for EXP_NAME=$EXP_NAME"
        echo "Config: $CONFIG"
        echo "Dataset: $DATASET_PATH"
        echo "View index: $VIEW_IDX_FILE"
        echo "--------------------------------------------"

        CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 1 --nnodes 1 \
          --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:39508 \
          inference.py \
          --config "$CONFIG" \
          training.checkpoint_dir="${CKPT_ROOT}/${EXP_NAME}/" \
          training.dataset_path="$DATASET_PATH" \
          training.target_has_input=false \
          training.square_crop=true \
          training.num_input_views="${V_INPUT}" \
          training.num_target_views="${V_TARGET}" \
          training.num_views="${V_TOTAL}" \
          inference.if_inference=true \
          inference.compute_metrics=true \
          inference.render_video=$RENDER_VIDEO \
          inference.render_video_config.num_frames=30 \
          inference.view_idx_file_path="$VIEW_IDX_FILE" \
          model.target_pose_tokenizer.patch_size="${PATCH_SIZE}" \
          training.batch_size_per_gpu=1 \
          inference_out_dir="${OUT_ROOT}/${EXP_NAME}/i${V_INPUT}_t${V_TARGET}_${DIFFICULTY}"

        echo "✅ Done: ${EXP_NAME} (V_INPUT=${V_INPUT}, V_TARGET=${V_TARGET})"
        echo "--------------------------------------------"
      done
    done
  done
done