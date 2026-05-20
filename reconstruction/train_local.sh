export HCCL_BUFFSIZE=120
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_STATS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export USE_CHECKPOINT=True
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_LAUNCH_BLOCKING=1
torchrun \
    --standalone \
    --nproc_per_node npu \
    train_ds_mixSup.py \
    --train_url outputs/debug_pi3 \
    --config configs/reliev3r_full_v8_c32_s1_dinov2_1B_Pi3/main.yaml \
    --ckpt_path /efs-mi-east4-2/checkpoints/Pi3 \
    --no_load_optim \
    --iterations 300000 \
    --backbone_lr_ratio 0.01 \
    --deepspeed \
    --deepspeed_config ./ds_config/ds_config_z2_ot.json