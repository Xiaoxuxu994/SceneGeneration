local_dir=.
token=2a9c4b51-96d0-41cd-a1b2-ecfa96c91de7
region=cn-east-3
bucket_name=bucket-multimodal
s3_valid=Spatial_Intelligence/Virtual_KITTI/
bkt_id=b7cb9493-708c-4f30-8aae-a6ac31ebae3d

python s3_downloader.py --objects_storage_path=$local_dir --app_token=$token --region=$region --bucket_name=$bucket_name --bucket_path=$s3_valid --bucket_id=$bkt_id --shared --show_speed
