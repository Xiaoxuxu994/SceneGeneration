# export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=2
# export ASCEND_CUSTOM_OPP_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_fit.py \
    --config_path configs/GSDecoder/exp_11_19_replica.json \
    --output_dir ../outputs/ \
    --opt models.voxel_align_ratio 1 \
    --select_scene room_2_2 \
    --opt trainer.eval_every 1 \
    --opt trainer.ckpt_every 0 \
    --opt trainer.lr_scheduler.args.warmup_steps 100 \
    --opt trainer.lr_scheduler.args.total_steps 5000 \    
    # --opt models.args.bp_acc $2 \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')