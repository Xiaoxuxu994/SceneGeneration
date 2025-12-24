# export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=1
# export ASCEND_CUSTOM_OPP_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node npu \
    trainer_torchrun.py $@ \
    --config_path configs/GSDecoder/exp_11_10_roma.json \
    --output_dir ../outputs/ \
    --opt pts_dir /efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole/geometry_priors/irregular_pts_align_unipts_normV14/mapanything_pcd_normext_imgonly_edgepreserve_csv_v14/npy \
    --opt depth_dir /efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole/geometry_priors/irregular_pts_align_unipts_normV14/mapanything_pcd_normext_imgonly_edgepreserve_csv_v14/depth_pred \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }')