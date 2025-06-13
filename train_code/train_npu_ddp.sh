ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_npu_col.py \
    --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py