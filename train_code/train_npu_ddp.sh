# ASCEND_LAUNCH_BLOCKING=1 ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col.py \
#     --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py

# ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col_ctf_new_metric.py \
#     --config configs/train/exp0620_3D_geometry_vae_normv6_ctf_4level_debug.py

ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col_ctf_new_metric_KL.py \
    --config configs/train/exp0721_3DGeoVAE_normv14_CTF_lr1e-4_shufflepoints.py
