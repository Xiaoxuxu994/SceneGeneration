# python python/cube_to_mesh.py \
#     --mode udf \
#     --engine mc \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --voxel_res 512 \
#     --num_workers 1 \
#     --level 0.0025 \
#     --force

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

python python/octree_to_mesh.py \
    --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree/ \
    --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree_mc/ \
    --src_size 256 \
    --tar_size 512 \
    --thres 3 \
    --mc_alpha 1 \
    --num_workers 1 \
    --force