export ASCEND_RT_VISIBLE_DEVICES=0
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_test_checklpipsnpu.py $@ \
    --config_path configs/GSDecoder/NPU_GS_decoder_mapanything_finetune_32views.json \
    --output_dir /efs-mi-east4-2/z00912387/1a444d034108/SceneGeneration/output/test/ \
    --opt ckpt ckpts/old/epoch_0290.ckpt \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')