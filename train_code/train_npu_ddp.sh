ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nproc_per_node=4 --master-port 29503 train_npu_col.py \
    --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py > train.log 2>&1 &