# run this script in the project root folder

local_dir=/mnt/nvme2/share/chenyouyu/codespace/vggt_sz/
BUCKET_PATH=y00855328/output/251201/agi/train/runs/2025-12-01_08-57-58/checkpoints/epoch036-loss_valid0.1726-metric_valid0.1199.ckpt/
APP_TOKEN=7e9dedd1-c231-cb5c-d86a-471869623324
REGION=cn-east-4
BUCKET_NAME=bucket-beijing-graphics
BUCKET_ID=38b6186c-32f0-445a-b2d2-7abb342ef9ff

python ./roma/s3_downloader.py \
    --objects_storage_path=$local_dir \
    --app_token=$APP_TOKEN \
    --region=$REGION \
    --bucket_name=$BUCKET_NAME \
    --bucket_path=$BUCKET_PATH \
    --bucket_id=$BUCKET_ID \
    --shared --show_speed