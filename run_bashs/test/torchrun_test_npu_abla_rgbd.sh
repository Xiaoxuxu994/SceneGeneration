export ASCEND_RT_VISIBLE_DEVICES=$2
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_test.py \
    --config_path configs/GSDecoder/opt_config.json \
    --output_dir outputs/$1/ \
    --opt ckpt ckpts/epoch_$1.ckpt \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')