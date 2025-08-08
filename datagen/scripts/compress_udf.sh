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
#     --thres 10 \
#     --num_workers 32 \
#     --check \
#     --stats

python python/compress_cube.py \
    --in_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/udf/ \
    --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/octree/ \
    --src_size 256 \
    --tar_size 512 \
    --thres 10 \
    --num_workers 32 \
    --check \
    --stats

python python/compress_cube.py \
    --in_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/udf/ \
    --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/octree/ \
    --src_size 256 \
    --tar_size 512 \
    --thres 10 \
    --num_workers 32 \
    --check \
    --stats