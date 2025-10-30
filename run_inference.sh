#!/bin/bash

# List of experiment names
EXP_NAMES=(
  # "LVSM_scene_dual_encoder_ftDecTok_lr25e-7"
  # "LVSM_scene_MapAnything_decoder_ftDecTok_ft_lr25e-7"
  "LVSM_scene_MapAnything_decoder_ftDecTok_iX"
  # "LVSM_scene_dual_encoder_ft_lr25e-7_scannetpp"
  "LVSM_scene_dual_encoder_ftDecTok_lr25e-7_i8"
  "LVSM_scene_dual_encoder_ftDecTok_lr25e-7_i8_scannetpp"
)

# Default list of input view counts
V_INPUTS=(2 32 4 8 16)

for EXP_NAME in "${EXP_NAMES[@]}"; do

  # If experiment name contains "scannetpp", add (V_INPUT=32, V_TARGET=32)
  if [[ "$EXP_NAME" == *"scannetpp"* ]]; then
    EXTRA_CASE=true
  else
    EXTRA_CASE=false
  fi

  for V_INPUT in "${V_INPUTS[@]}"; do
    V_TARGET=3
    PATCH_SIZE=8
    V_TOTAL=$((V_TARGET + V_INPUT))

    # ------------------------
    # 1. Choose CONFIG adaptively
    # ------------------------
    if [[ "$EXP_NAME" == *"scannetpp"* && "$EXP_NAME" == *"dual"* ]]; then
      CONFIG="configs/LVSM_scene_dual_encoder_scannetpp.yaml"
    elif [[ "$EXP_NAME" == *"dual"* ]]; then
      CONFIG="configs/LVSM_scene_dual_encoder.yaml"
    elif [[ "$EXP_NAME" == *"scannetpp"* ]]; then
      CONFIG="configs/LVSM_scene_MapAnything_decoder_scannetpp.yaml"
    else
      CONFIG="configs/LVSM_scene_MapAnything_decoder.yaml"
    fi

    # ------------------------
    # 2. Choose DATASET_PATH adaptively
    # ------------------------
    if [[ "$EXP_NAME" == *"scannetpp"* ]]; then
      DATASET_PATH="/storage/datasets/nargenziano/scannetpp/splits/hq_lvsm_val.txt"
    else
      DATASET_PATH="/storage/datasets/nargenziano/re10k_LVSM/test/full_list.txt"
    fi

    # ------------------------
    # 3. Choose VIEW_IDX_FILE adaptively
    # ------------------------
    if [[ "$EXP_NAME" == *"scannetpp"* ]]; then
      VIEW_IDX_FILE="data/scannetpp/evaluation_index_scannetpp_i${V_INPUT}_t${V_TARGET}.json"
    else
      VIEW_IDX_FILE="data/re10k/evaluation_index_re10k_i${V_INPUT}_t${V_TARGET}.json"
    fi

    # ------------------------
    # 4. Set batch size only for scannetpp
    # ------------------------
    if [[ "$EXP_NAME" == *"scannetpp"* ]]; then
      BATCH_SIZE_ARG="training.batch_size_per_gpu=1"
    else
      BATCH_SIZE_ARG=""
    fi

    echo "Running inference for EXP_NAME=$EXP_NAME"
    echo "Config: $CONFIG"
    echo "Dataset: $DATASET_PATH"
    echo "View index: $VIEW_IDX_FILE"
    echo "Batch size arg: ${BATCH_SIZE_ARG:-default}"
    echo "--------------------------------------------"

    CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 1 --nnodes 1 \
      --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:39507 \
      inference.py \
      --config "$CONFIG" \
      training.checkpoint_dir="/storage/home/nargenziano/LVSM/checkpoints/${EXP_NAME}/" \
      training.dataset_path="$DATASET_PATH" \
      training.target_has_input=false \
      training.square_crop=true \
      training.num_input_views="${V_INPUT}" \
      training.num_target_views="${V_TARGET}" \
      training.num_views="${V_TOTAL}" \
      inference.if_inference=true \
      inference.compute_metrics=true \
      inference.render_video=false \
      inference.view_idx_file_path="$VIEW_IDX_FILE" \
      model.target_pose_tokenizer.patch_size="${PATCH_SIZE}" \
      $BATCH_SIZE_ARG \
      inference_out_dir="/storage/home/nargenziano/LVSM/experiments/evaluation/${EXP_NAME}/ii${V_INPUT}_it${V_TARGET}"

    echo "✅ Done: ${EXP_NAME} (V_INPUT=${V_INPUT}, V_TARGET=${V_TARGET})"
    echo "--------------------------------------------"
  done

  # ------------------------
  # Extra case for scannetpp: V_INPUT=32, V_TARGET=32
  # ------------------------
  if $EXTRA_CASE; then
    V_INPUT=32
    V_TARGET=32
    V_TOTAL=$((V_INPUT + V_TARGET))

    echo "Running extra inference case for EXP_NAME=$EXP_NAME (scannetpp only) with V_INPUT=$V_INPUT, V_TARGET=$V_TARGET"

    CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 1 --nnodes 1 \
      --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:39507 \
      inference.py \
      --config "$CONFIG" \
      training.checkpoint_dir="/storage/home/nargenziano/LVSM/checkpoints/${EXP_NAME}/" \
      training.dataset_path="$DATASET_PATH" \
      training.target_has_input=false \
      training.square_crop=true \
      training.num_input_views="${V_INPUT}" \
      training.num_target_views="${V_TARGET}" \
      training.num_views="${V_TOTAL}" \
      inference.if_inference=true \
      inference.compute_metrics=true \
      inference.render_video=true \
      inference.render_video_config.num_frames=10 \
      inference.view_idx_file_path="data/scannetpp/evaluation_index_scannetpp_i${V_INPUT}_t${V_TARGET}.json" \
      model.target_pose_tokenizer.patch_size="${PATCH_SIZE}" \
      training.batch_size_per_gpu=1 \
      inference_out_dir="/storage/home/nargenziano/LVSM/experiments/evaluation/${EXP_NAME}/ii${V_INPUT}_it${V_TARGET}"

    echo "✅ Done (extra scannetpp case): ${EXP_NAME} (V_INPUT=${V_INPUT}, V_TARGET=${V_TARGET})"
    echo "--------------------------------------------"
  fi

done