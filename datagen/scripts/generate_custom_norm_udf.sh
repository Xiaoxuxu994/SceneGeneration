# python python/export_cube.py \
#     --mode udf \
#     --dataset custom \
#     --in_dir data/datasets/riemann_mesh/20250821/ \
#     --norm_dir data/datasets/riemann_mesh_-1_1_0.95/20250821/ \
#     --out_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --voxel_res 256 \
#     --num_workers 1

# python python/export_cube.py \
#     --mode udf \
#     --dataset custom \
#     --in_dir data/datasets/riemann_mesh/20250821/ \
#     --norm_dir data/datasets/riemann_mesh_-1_1_0.95/20250821/ \
#     --out_dir data/datasets/riemann_volume_-1_1_0.95/20250821/ \
#     --voxel_res 512 \
#     --num_workers 1

python python/export_cube.py \
    --mode udf \
    --dataset custom \
    --in_dir data/datasets/riemann_scan_mesh/ \
    --norm_dir data/datasets/riemann_scan_mesh_-1_1_0.95/ \
    --out_dir data/datasets/riemann_scan_volume_-1_1_0.95/ \
    --voxel_res 256 \
    --num_workers 2

python python/export_cube.py \
    --mode udf \
    --dataset custom \
    --in_dir data/datasets/riemann_scan_mesh/ \
    --norm_dir data/datasets/riemann_scan_mesh_-1_1_0.95/ \
    --out_dir data/datasets/riemann_scan_volume_-1_1_0.95/ \
    --voxel_res 512 \
    --num_workers 2