# run this script in the project root folder

local_dir=/mnt/nvme1/share/chenzhenyu/
BUCKET_PATH=y00855328/sr_datasets/processed/7scenes/data/chess/seq-01/
APP_TOKEN=7e9dedd1-c231-cb5c-d86a-471869623324
REGION=cn-east-4
BUCKET_NAME=bucket-beijing-graphics
BUCKET_ID=63d208b2-9ff7-45cf-a22f-6f0c32656425

python ./roma/s3_downloader.py \
    --objects_storage_path=$local_dir \
    --app_token=$APP_TOKEN \
    --region=$REGION \
    --bucket_name=$BUCKET_NAME \
    --bucket_path=$BUCKET_PATH \
    --bucket_id=$BUCKET_ID \
    --shared --show_speed