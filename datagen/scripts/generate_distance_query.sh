# python python/generate_distance_query.py \
#     --in_dir data/datasets/scannetpp_mesh_norm_-1_1_0.95/ \
#     --out_dir data/datasets/scannetpp_volume_norm_-1_1_0.95/udf/ \
#     --num_workers 1 \
#     --force

python python/generate_distance_query.py \
    --in_dir data/efs-mi-east4-2/datasets/arkitscenes_processed/arkitscenes_mesh_norm_-1_1_0.95/train/ \
    --out_dir data/efs-mi-east4-2/datasets/arkitscenes_processed/arkitscenes_volume_norm_-1_1_0.95/train/udf/ \
    --num_workers 1 

# python python/generate_distance_query.py \
#     --in_dir data/nvme2/share/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
#     --out_dir data/efs-mi-east4-2/datasets/arkitscenes_processed/arkitscenes_volume_norm_-1_1_0.95/val/udf/ \
#     --num_workers 1 