# export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=7
# export ASCEND_CUSTOM_OPP_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port 22222 \
    --mixed_precision bf16 \
    --num_processes 1 \
    trainer.py \
    --config_path configs/GSDecoder/debug.json \
    --output_dir /mnt/nvme/yyx/outputs \
    --use_depth_mask \
    --select_scene 8013901416 \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')
