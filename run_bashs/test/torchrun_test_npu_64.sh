export ASCEND_RT_VISIBLE_DEVICES=3
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_test.py $@ \
    --config_path configs/GSDecoder/NPU_GS_decoder_mapanything_finetune_64views.json \
    --output_dir /efs-mi-east4-2/z00912387/1a444d034108/SceneGeneration/output/64ckpt_64v/ \
    --opt ckpt ckpts/decoder/64views_epoch_0390.ckpt \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')