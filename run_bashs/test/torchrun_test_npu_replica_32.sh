export ASCEND_RT_VISIBLE_DEVICES=0
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_fit_test.py $@ \
    --config_path configs/GSDecoder/exp_11_17_replica_test_32.json \
    --output_dir /efs-mi-east4-2/z84301856/AnchorSplat/output/test/ \
    --opt ckpt ckpts/decoder/replica_epoch_2500.ckpt \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')
