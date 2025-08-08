# python python/export_cube.py \
#     --mode sdf \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 16 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode sdf \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 32 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode sdf \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 64 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode sdf \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 128 \
#     --num_workers 20

python python/export_cube.py \
    --mode sdf \
    --dataset scannetpp \
    --in_dir data/datasets/scannetpp_mesh/data/ \
    --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
    --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
    --voxel_res 256 \
    --num_workers 16 \
    --is_norm

# python python/export_cube.py \
#     --mode sdf \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 512 \
#     --num_workers 20