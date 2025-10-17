# # DEBUG 
# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Training/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/ \
#     --voxel_res 512 \
#     --num_workers 1

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Training/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/ \
#     --voxel_res 16 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Training/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/ \
#     --voxel_res 32 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Training/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/ \
#     --voxel_res 64 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Training/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/ \
#     --voxel_res 128 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir data/nvme2/share/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --is_norm \
#     --norm_dir data/nvme2/share/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --out_dir data/nvme2/share/datasets/arkitscenes_volume_norm_-1_1_0.95/train/ \
#     --voxel_res 256 \
#     --num_workers 20

python python/export_cube.py \
    --mode udf \
    --dataset arkitscenes \
    --in_dir data/nvme2/share/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
    --is_norm \
    --norm_dir data/nvme2/share/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
    --out_dir data/nvme2/share/datasets/arkitscenes_volume_norm_-1_1_0.95/val/ \
    --voxel_res 256 \
    --num_workers 16

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Training/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/train/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/train/ \
#     --voxel_res 512 \
#     --num_workers 10

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Validation/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/ \
#     --voxel_res 16 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Validation/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/ \
#     --voxel_res 32 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Validation/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/ \
#     --voxel_res 64 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Validation/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/ \
#     --voxel_res 128 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Validation/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/ \
#     --voxel_res 256 \
#     --num_workers 20

# python python/export_cube.py \
#     --mode udf \
#     --dataset arkitscenes \
#     --in_dir /efs-large-spatial-model/dataset/ARKitScene/raw/Validation/ \
#     --norm_dir data/datasets/arkitscenes_mesh_norm_-1_1_0.95/val/ \
#     --out_dir data/datasets/arkitscenes_volume_norm_-1_1_0.95/val/ \
#     --voxel_res 512 \
#     --num_workers 10