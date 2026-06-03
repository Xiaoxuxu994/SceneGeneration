# run this script in the project root folder

LOCAL_FOLDER_ABS_PATH=/efs-mi-east4-2/checkpoints/VGGT-1B/
BUCKET_PATH=c00894105/checkpoints/VGGT-1B/
APP_TOKEN=7e9dedd1-c231-cb5c-d86a-471869623324
REGION=cn-east-4
BUCKET_NAME=bucket-mi-east4-sf
BUCKET_ID=63d208b2-9ff7-45cf-a22f-6f0c32656425

python ./roma/s3_uploader.py \
    --local_folder_absolute_path=$LOCAL_FOLDER_ABS_PATH \
    --app_token=$APP_TOKEN \
    --region=$REGION \
    --bucket_name=$BUCKET_NAME \
    --bucket_path=$BUCKET_PATH \
    --bucket_id=$BUCKET_ID \
    --shared --show_speed