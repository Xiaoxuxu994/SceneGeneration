ASCEND_RT_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node npu \
    train_mixSup.py \
    --train_url outputs/debug \
    --config configs/reliev3r_full_v8_c32_s1_dinov2_1B_camRegTrue/main.yaml \
    --ckpt_path /efs-mi-east4-3/c50041401/exp/ssff3d/Reliev3R-260424-mixSupervise-lr1e-5-SV-SS-Pi3-freeze-resume-9/ckpt/70000.ckpt \
    --backbone_lr_ratio .01 \
    --no_load_optim \
    --iterations 300000 \
    --deepspeed \
    --deepspeed_config ./ds_config/ds_config_z2_ot.json