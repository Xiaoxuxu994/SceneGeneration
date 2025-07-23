# ASCEND_RT_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 --master-port 29540 train_npu_col.py \
#     --config configs/exp0701_3D_geometry_vae_releaseV1_train.py > train.log 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nproc_per_node=4 --master-port 29541 train_npu_new_metric.py \
#     --config configs/exp0612_3D_geometry_vae_releaseV1_new_metric.py > train_async.log 2>&1 &

# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29541 train_npu_new_metric.py \
#     --config configs/exp0612_3D_geometry_vae_releaseV1_new_metric.py
# ASCEND_LAUNCH_BLOCKING=1 ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col.py \
#     --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py

# ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col_ctf_new_metric.py \
#     --config configs/train/exp0620_3D_geometry_vae_normv6_ctf_4level_debug.py

ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col_ctf_new_metric_KL.py \
    --config configs/train/exp0721_3DGeoVAE_normv14_CTF_lr1e-4.py
