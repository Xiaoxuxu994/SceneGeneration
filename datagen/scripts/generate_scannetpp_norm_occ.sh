python python/export_cube.py \
    --mode occ \
    --in_dir data/datasets/scannetpp_mesh/data/ \
    --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
    --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
    --voxel_res 128 \
    --num_workers 1 \
    --force

# python python/export_cube.py \
#     --mode occ \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 128 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode occ \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 256 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode occ \
#     --in_dir data/datasets/scannetpp_mesh/data/ \
#     --norm_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95 \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95 \
#     --voxel_res 512 \
#     --num_workers 20