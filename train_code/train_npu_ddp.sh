ASCEND_RT_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 --master-port 29540 train_npu_col.py \
    --config configs/exp0701_3D_geometry_vae_releaseV1_train.py > train.log 2>&1 &
