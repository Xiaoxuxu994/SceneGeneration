rm -rf kernel_meta fusion_result.json outputs
find /home/yyx/SceneGeneration/reconstruction -type d -name '__pycache__' -exec rm -rf {} +
python /home/yyx/s3_uploader.py \
    --local_folder_absolute_path=/home/yyx/SceneGeneration/reconstruction \
    --app_token=7e9dedd1-c231-cb5c-d86a-471869623324 \
    --region=cn-east-4 \
    --bucket_name=bucket-mi-east4-sf \
    --bucket_path=y00959808/Reliev3R-mixsup/ \
    --show_speed