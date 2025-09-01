# # MeshUDF
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --grad_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine meshudf \
#     --precise_grad \
#     --num_workers 1 \
#     --force

# # CapUDF
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --grad_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine capudf \
#     --precise_grad \
#     --level 0.003 \
#     --num_workers 1 \
#     --force

# # Marching Cubes on octree
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine mc \
#     --level 0.003 \
#     --num_workers 1 \
#     --force

# Marching Cubes on original volume
python python/cube_to_mesh.py \
    --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
    --mode udf \
    --voxel_res 512 \
    --engine mc \
    --level 0.003 \
    --num_workers 1 \
    --force