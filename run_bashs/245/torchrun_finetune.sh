export ASCEND_RT_VISIBLE_DEVICES=2,3
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun.py \
    --config_path configs/GSDecoder/exp_11_13_finetune.json \
    --output_dir ../outputs \
    --opt pts_dir /efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole/geometry_priors/irregular_pts_align_unipts_normV14/mapanything_pcd_normext_imgonly_64view_v2_edgepreserve_csv/mapanything_imgonly_v2_${1}view/ply_pred \
    --opt depth_dir /efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole/geometry_priors/irregular_pts_align_unipts_normV14/mapanything_pcd_normext_imgonly_64view_v2_edgepreserve_csv/mapanything_imgonly_v2_${1}view/depth_pred \
    --opt view_select $1 \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')