# run this script in the project root folder

# local_dir=/mnt/nvme2/share/chenzhenyu/reconstruction_benchmark/
# BUCKET_PATH=y00855328/sr_datasets/processed/kitti/data/raw_data/2011_10_03/2011_10_03_drive_0027_sync/image_03/
# APP_TOKEN=7e9dedd1-c231-cb5c-d86a-471869623324
# REGION=cn-east-4
# BUCKET_NAME=bucket-beijing-graphics
# BUCKET_ID=38b6186c-32f0-445a-b2d2-7abb342ef9ff

local_dir=/mnt/nvme2/share/chenzhenyu/reconstruction_benchmark/
BUCKET_PATH=dataset/dust3r/MPI-Sintel/
APP_TOKEN=7e9dedd1-c231-cb5c-d86a-471869623324
REGION=cn-southwest-2
BUCKET_NAME=bucket-471
BUCKET_ID=713110d9-7270-4ae6-aeea-21e797076a46

python ./roma/s3_downloader.py \
    --objects_storage_path=$local_dir \
    --app_token=$APP_TOKEN \
    --region=$REGION \
    --bucket_name=$BUCKET_NAME \
    --bucket_path=$BUCKET_PATH \
    --bucket_id=$BUCKET_ID \
    --shared --show_speed