# ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col.py \
#     --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 train_npu_new_metric.py \
    --config configs/exp0612_3D_geometry_vae_releaseV1.py