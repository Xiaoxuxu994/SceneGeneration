# export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=7
# export ASCEND_CUSTOM_OPP_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun.py \
    --config_path configs/GSDecoder/exp_11_07_base.json \
    --output_dir ../outputs/ \
    --use_depth_mask \
    --select_scene 8013901416 \
    --opt viewInfo_dir /home/yyx/data/debug_data/v14 \
    --opt pts_dir /home/yyx/data/debug_data/map_anything \
    --opt trainer.eval_every 1 \
    --opt trainer.ckpt_every 0 \
    --opt trainer.lr_scheduler.args.warmup_steps 20 \
    --opt trainer.lr_scheduler.args.total_steps 500 \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')