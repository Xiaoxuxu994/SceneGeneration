LOCAL_FOLDER_ABS_PATH=/efs-mi-east4-2/datasets/DL3DV/matching_preprocess
BUCKET_PATH=c50041401/dataset/DL3DV/
python s3_uploader.py --local_folder_absolute_path=$LOCAL_FOLDER_ABS_PATH --app_token=7e9dedd1-c231-cb5c-d86a-471869623324 --region=cn-east-4 --bucket_name=bucket-yellow-ci-hd-turing --bucket_path=$BUCKET_PATH --shared --show_speed --bucket_id=63d208b2-9ff7-45cf-a22f-6f0c32656425
