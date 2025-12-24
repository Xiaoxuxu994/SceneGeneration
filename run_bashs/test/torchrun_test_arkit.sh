export ASCEND_RT_VISIBLE_DEVICES=5
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_test.py \
    --config_path configs/Refiner/exp_11_20_arkit_32_test.json \
    --output_dir /mnt/nvme3/yyx/outputs/arkit/refiner/ \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')