# export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
# export ASCEND_CUSTOM_OPP_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun.py $@ \
    --config_path configs/GSDecoder/exp_12_09_ablation_ma_depth_online_anchor_v4.json \
    --output_dir ../outputs/ \
    # --opt viewInfo_dir /home/yyx/code/anchor \
    # --opt depth_dir /home/yyx/code/anchor \
    # --opt trainer.max_epochs 1 \
    # --profile \
    # --select_scene 31a2c91c43 \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')