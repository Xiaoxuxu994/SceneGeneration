"""
    Used for downloading benchmark sample data from S3, where each dataset
    only includes one sequence.
    The folder structures are kept identical to the original ones.
"""

import subprocess

from pathlib import Path

CMD = "python ./roma/s3_downloader.py \
    --objects_storage_path={} \
    --bucket_path={} \
    --app_token=7e9dedd1-c231-cb5c-d86a-471869623324 \
    --region=cn-east-4 \
    --bucket_name=bucket-beijing-graphics \
    --bucket_id=63d208b2-9ff7-45cf-a22f-6f0c32656425 \
    --shared --show_speed"

SRC_ROOT = "y00855328/sr_datasets/processed/"
SRC_SUBTREE = [
    "7scenes/data/chess/seq-01/",
    "DTU/spann3r_version/dtu_test_mvsnet_release/scan1/",
    "ETH3D/multi_view_training_dslr/courtyard/",
    "NRGBD/data/breakfast_room/",
    "scannet/posed_images/scene0707_00/",
    "tanks_and_temples/data/Barn/",
    "kitti/data/depth_completion_prediction/val/2011_09_30_drive_0016_sync/"
]
TAR_ROOT = Path("./samples/")

for src_subtree in SRC_SUBTREE:
    src_tree = SRC_ROOT + src_subtree
    tar_tree = TAR_ROOT / src_subtree
    tar_tree = tar_tree.parent
    tar_tree.mkdir(parents=True, exist_ok=True)
    
    print("downloading {} to {}".format(src_tree, str(tar_tree)))
    
    cmd = CMD.format(str(tar_tree), src_tree)
    _ = subprocess.call(cmd.split(' '))
    

