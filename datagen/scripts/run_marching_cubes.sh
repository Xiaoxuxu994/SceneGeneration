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

# # MeshUDF
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --grad_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine meshudf \
#     --precise_grad \
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

# # Marching Cubes on octree
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine mc \
#     --level 0.003 \
#     --num_workers 1 \
#     --force

# # Marching Cubes on octree
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_scan_volume_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine mc \
#     --level 0.003 \
#     --num_workers 1 \
#     --force

# # Marching Cubes on octree
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95_example/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine mc \
#     --level 0.003 \
#     --num_workers 1 \
#     --force

# # Marching Cubes on original volume
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --mode udf \
#     --voxel_res 512 \
#     --engine mc \
#     --level 0.003 \
#     --num_workers 1 \
#     --force

# # XYZ point cloud
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --grad_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --sign_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine pcd \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --num_workers 1 \
#     --force

# # XYZ point cloud
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --grad_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --sign_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine pcd \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --num_workers 1 \
#     --force

# # XYZ point cloud
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_sample_volume_-1_1_0.95/ \
#     --grad_dir data/datasets/scannetpp_sample_volume_-1_1_0.95/ \
#     --sign_dir data/datasets/scannetpp_sample_volume_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 1 \
#     --engine pcd \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --num_workers 1 \
#     --force

# # Poisson Reconstruction
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --grad_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --sign_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine poisson \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --trim \
#     --num_workers 1 \
#     --force

# # Poisson Reconstruction
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_sample_volume_-1_1_0.95/ \
#     --grad_dir data/datasets/scannetpp_sample_volume_-1_1_0.95/ \
#     --sign_dir data/datasets/scannetpp_sample_volume_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 1 \
#     --engine poisson \
#     --depth 10 \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --trim \
#     --num_workers 1 \
#     --force

# Poisson Reconstruction using compiled executables
python python/cube_to_mesh.py \
    --in_dir data/datasets/scannetpp_results/20250902/ \
    --grad_dir data/datasets/scannetpp_results/20250902/ \
    --sign_dir data/datasets/scannetpp_results/20250902/ \
    --mode octree \
    --voxel_res 512 \
    --oct_src_size 256 \
    --oct_tar_size 512 \
    --oct_thres 3 \
    --engine poisson_exe \
    --trim \
    --depth 10 \
    --suffix ply \
    --precise_grad \
    --normal \
    --num_workers 1 \
    --force

# # Poisson Reconstruction
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --grad_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --sign_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine poisson \
#     --depth 9 \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --num_workers 1 \
#     --force

# # Poisson Reconstruction
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95_example/ \
#     --grad_dir data/datasets/scannetpp_volume_norm_-1_1_0.95_example/ \
#     --sign_dir data/datasets/scannetpp_volume_norm_-1_1_0.95_example/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine poisson \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --num_workers 1 \
#     --force

# # Poisson Reconstruction
# python python/cube_to_mesh.py \
#     --in_dir data/datasets/riemann_scan_volume_-1_1_0.95/ \
#     --grad_dir data/datasets/riemann_scan_volume_-1_1_0.95/ \
#     --sign_dir data/datasets/riemann_scan_volume_-1_1_0.95/ \
#     --mode octree \
#     --voxel_res 512 \
#     --oct_src_size 256 \
#     --oct_tar_size 512 \
#     --oct_thres 3 \
#     --engine poisson \
#     --depth 10 \
#     --suffix ply \
#     --precise_grad \
#     --normal \
#     --num_workers 1 \
#     --force