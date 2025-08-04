# python python/compress_cube.py \
#     --in_dir data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/ \
#     --out_dir data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 5 \
#     --num_workers 1 \
#     --force \
#     --check

# python python/compress_cube.py \
#     --in_dir data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/ \
#     --out_dir data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/ \
#     --src_size 256 \
#     --tar_size 512 \
#     --thres 5 \
#     --num_workers 32 \
#     --check

python python/compress_cube.py \
    --in_dir data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/ \
    --out_dir data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/ \
    --out_dir_sec data/nvme2/scannetpp_volume_norm_-1_1_0.95/octree/ \
    --src_size 256 \
    --tar_size 512 \
    --thres 5 \
    --num_workers 1 \
    --stats