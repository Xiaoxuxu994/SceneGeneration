ASCEND_RT_VISIBLE_DEVICES=4 torchrun \
    --standalone \
    --nproc_per_node npu \
    train_mixSup.py \
    --train_url outputs/debug \
    --config configs/reliev3r_full_v8_c32_s1_dinov2_1B/main.yaml \
    --ckpt_path /efs-mi-east4-2/checkpoints/Pi3 \
    --backbone_lr_ratio 0.01 \
    --no_load_optim \
    --iterations 300000 \
    # --deepspeed \
    # --deepspeed_config ./ds_config/ds_config_z2_ot.json