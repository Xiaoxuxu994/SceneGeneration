#!/bin/bash

# List of experiment names
EXP_NAMES=(
  # "LVSM_scene_MapAnything_decoder_lr25e-5_ftDecTok"
  # "LVSM_scene_MapAnything_decoder_lr25e-5_ftDec"
  # "LVSM_PRoPE_scene_MA_decoder"
  # "LVSM_scene_MapAnything_decoder_ftDecTok"
  # "LVSM_scene_MapAnything_decoder_ftDec"
  # "LVSM_scene_MapAnything_decoder_lr25e-5_tokenizer_nonorm"
  # "LVSM_scene_MapAnything_decoder_ftDecTok_i8"
  "LVSM_scene_MapAnything_decoder_ftDecTok_ft_lr25e-7"
  "LVSM_scene_MapAnything_decoder_ft_lr25e-5"
  # "LVSM_scene_MapAnything_decoder_ftre10k_scannetpp_i8"
  # "LVSM_scene_MapAnything_decoder_ftDecTok_scannetpp_i8"
  # "LVSM_scene_MapAnything_decoder_ftDecTok_scannetpp"
)

# List of input view counts
V_INPUTS=(2 32 4 8 16)
# V_INPUTS=(32)

for V_INPUT in "${V_INPUTS[@]}"; do
  for EXP_NAME in "${EXP_NAMES[@]}"; do

    # # Determine patch size based on EXP_NAME
    # if [[ "$EXP_NAME" == *"ftDecTok"* ]]; then
    #   PATCH_SIZE=8
    # else
    #   PATCH_SIZE=14
    # fi
    PATCH_SIZE=8

    V_TARGET=3
    V_TOTAL=$((V_TARGET + V_INPUT))

    echo "Running inference for EXP_NAME=$EXP_NAME with V_INPUT=$V_INPUT, V_TARGET=$V_TARGET, V_TOTAL=$V_TOTAL (patch_size=$PATCH_SIZE)"

    CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 1 --nnodes 1 \
      --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:39507 \
      inference.py \
      --config "configs/LVSM_scene_MapAnything_decoder.yaml" \
      training.checkpoint_dir="/storage/home/nargenziano/LVSM/checkpoints/${EXP_NAME}/" \
      training.dataset_path="/storage/datasets/nargenziano/re10k_LVSM/test/full_list.txt" \
      training.target_has_input=false \
      training.square_crop=true \
      training.num_input_views="${V_INPUT}" \
      training.num_target_views="${V_TARGET}" \
      training.num_views="${V_TOTAL}" \
      inference.if_inference=true \
      inference.compute_metrics=true \
      inference.render_video=false \
      inference.view_idx_file_path="data/re10k/evaluation_index_re10k_i${V_INPUT}_t${V_TARGET}.json" \
      model.target_pose_tokenizer.patch_size="${PATCH_SIZE}" \
      inference_out_dir="/storage/home/nargenziano/LVSM/experiments/evaluation/${EXP_NAME}/ii${V_INPUT}_it${V_TARGET}"

    echo "✅ Done: ${EXP_NAME} (V_INPUT=${V_INPUT}, V_TARGET=${V_TARGET})"
    echo "--------------------------------------------"
  done
done
