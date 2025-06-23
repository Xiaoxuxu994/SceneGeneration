ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col.py \
    --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py