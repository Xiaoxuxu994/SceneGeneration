export ASCEND_RT_VISIBLE_DEVICES=$2
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_test.py \
    --config_path configs/Refiner/exp_11_14_finetune_${1}_test.json \
    --output_dir /mnt/nvme3/yyx/outputs/refiner_2/ \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')