# export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=3
# export ASCEND_CUSTOM_OPP_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_fit.py \
    --config_path configs/GSDecoder/exp_11_19_arkitscenes.json \
    --output_dir ../outputs/ \
    --opt models.voxel_align_ratio $1 \
    # --opt models.args.bp_acc $2 \    
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')