python python/cube_to_mesh.py \
    --mode sdf \
    --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
    --voxel_res 256 \
    --num_workers 1 \
    --alpha 0.5 \
    --force

# python python/cube_to_mesh.py \
#     --mode udf \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 128 \
#     --num_workers 20

# python python/cube_to_mesh.py \
#     --mode udf \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 256 \
#     --num_workers 20

# python python/cube_to_mesh.py \
#     --mode udf \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 512 \
#     --num_workers 20

# python python/octree_to_mesh.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree/ \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree_mc/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 5 \
#     --num_workers 1 \
#     --force