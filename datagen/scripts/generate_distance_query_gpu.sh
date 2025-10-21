python python/generate_distance_query_gpu.py \
    --in_dir data/efs-mi-east4-2/datasets/arkitscenes_processed/arkitscenes_mesh_norm_-1_1_0.95/val/ \
    --out_dir data/efs-mi-east4-2/datasets/arkitscenes_processed/arkitscenes_volume_norm_-1_1_0.95/val/udf/ \
    --num_queries 1000000000 \
    --chunk_size 100000000
