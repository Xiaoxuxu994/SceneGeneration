# local_dir=.
# token=2a9c4b51-96d0-41cd-a1b2-ecfa96c91de7
# region=cn-east-3
# bucket_name=bucket-multimodal
# s3_valid=Spatial_Intelligence/Virtual_KITTI/
# bkt_id=b7cb9493-708c-4f30-8aae-a6ac31ebae3d

local_dir=/efs-m3ga/z00533370/reliev3r_data/
token=a3d8017e-5f91-48bf-8468-c60c6aa53d85
region=cn-southwest-2
bucket_name=bucket-2565
s3_valid=Spatial_Intelligence/MegaSynth/SceneNet-RGB-D/
bkt_id=923e457c-7281-4c1c-9ca3-b504f139a66d

python s3_downloader.py --objects_storage_path=$local_dir --app_token=$token --region=$region --bucket_name=$bucket_name --bucket_path=$s3_valid --bucket_id=$bkt_id --shared --show_speed
