# python python/compress_cube.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/udf/ \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree/ \
#     --mc_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree_mc/ \
#     --mc_alpha 1 \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 10 \
#     --num_workers 1 \
#     --check \
#     --force

# python python/compress_cube.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/udf/ \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 2 \
#     --num_workers 30 \
#     --check \
#     --stats

# python python/compress_cube.py \
#     --in_dir data/datasets/riemann_volume_-1_1_0.95/20250821/udf/ \
#     --out_dir data/datasets/riemann_volume_-1_1_0.95/20250821/octree/ \
#     --mesh_dir data/datasets/riemann_mesh_-1_1_0.95/20250821/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 3 \
#     --num_workers 1 \
#     --grad \
#     --sign \
#     --check \
#     --stats

# python python/compress_cube.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/udf/ \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree/ \
#     --mesh_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 3 \
#     --num_workers 1 \
#     --grad \
#     --sign \
#     --check \
#     --stats

# python python/compress_cube.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/udf/ \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree/ \
#     --mesh_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 3 \
#     --num_workers 32 \
#     --check \
#     --stats 

# python python/compress_cube.py \
#     --in_dir data/datasets/riemann_scan_volume_-1_1_0.95/udf/ \
#     --out_dir data/datasets/riemann_scan_volume_-1_1_0.95/octree/ \
#     --mesh_dir data/datasets/riemann_scan_mesh_-1_1_0.95/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 2 \
#     --num_workers 1 \
#     --grad \
#     --sign \
#     --check \
#     --stats

# python python/compress_cube.py \
#     --in_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/udf/ \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/octree/ \
#     --mesh_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 3 \
#     --num_workers 32 \
#     --check \
#     --stats

# python python/compress_cube.py \
#     --in_dir data/nvme2/share/datasets/arkitscenes_volume_norm_-1_1_0.95/train/udf/ \
#     --out_dir data/efs-mi-east4-2/datasets/arkitscenes_processed/arkitscenes_volume_norm_-1_1_0.95/train/octree/ \
#     --mesh_dir data/nvme2/share/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 3 \
#     --num_workers 8 \
#     --check

python python/compress_cube.py \
    --in_dir data/nvme2/share/datasets/arkitscenes_volume_norm_-1_1_0.95/val/udf/ \
    --out_dir data/efs-mi-east4-2/datasets/arkitscenes_processed/arkitscenes_volume_norm_-1_1_0.95/val/octree/ \
    --mesh_dir data/nvme2/share/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
    --src_size 256 \
    --tar_size 512 \
    --thres 3 \
    --num_workers 8 \
    --check