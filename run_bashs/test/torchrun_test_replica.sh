export ASCEND_RT_VISIBLE_DEVICES=4
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_test.py \
    --config_path configs/Refiner/exp_11_20_replica_32_test.json \
    --output_dir /mnt/nvme3/yyx/outputs/replica/refiner/plus${1}/ \
    --opt NVviewInfo_dir /efs-mi-east4-2/datasets/replica_processed_full/novel_view_fix8_plus${1}_viewInfo_normV14 \
    --opt NVdepth_dir /efs-mi-east4-2/datasets/replica_processed_full/mapanything_imgonly_v2_32view/depth_pred_nv4_plus${1} \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')