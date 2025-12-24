export ASCEND_RT_VISIBLE_DEVICES=3
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun_test.py $@ \
    --config_path configs/GSDecoder/NPU_GS_decoder_mapanything_finetune_32views_gs8.json \
    --output_dir /efs-mi-east4-2/z00912387/1a444d034108/SceneGeneration/output/abla_gs8/ \
    --opt ckpt ckpts/ablation/gs8/epoch_0130.ckpt \
    --test \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')