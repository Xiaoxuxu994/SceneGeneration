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
    --bucket_id=38b6186c-32f0-445a-b2d2-7abb342ef9ff \
    --shared --show_speed"

# SRC_ROOT = "y00855328/output/251201/agi/train/runs/2025-12-01_08-57-58/checkpoints/"
# SRC_SUBTREE = [
#     # "epoch009-loss_valid0.2741-metric_valid0.2231.ckpt/",
#     # "epoch011-loss_valid0.2917-metric_valid0.2021.ckpt/",
#     # "epoch013-loss_valid0.2557-metric_valid0.1859.ckpt/",
#     # "epoch015-loss_valid0.2459-metric_valid0.1737.ckpt/",
#     # "epoch017-loss_valid0.2416-metric_valid0.1665.ckpt/",
#     # "epoch019-loss_valid0.2314-metric_valid0.1583.ckpt/",
#     # "epoch021-loss_valid0.2167-metric_valid0.1511.ckpt/",
#     # "epoch023-loss_valid0.2258-metric_valid0.1446.ckpt/"
#     "epoch036-loss_valid0.1726-metric_valid0.1199.ckpt/",
#     "epoch056-loss_valid0.1663-metric_valid0.0993.ckpt/"
# ]
# TAR_ROOT = Path("/mnt/nvme2/share/chenyouyu/codespace/vggt_sz/")

SRC_ROOT = "y00855328/output/Ver.10-FT/mi/train/runs/2026-01-16_15-55-31/checkpoints/"
SRC_SUBTREE = [
    "epoch021-loss_valid0.1170-metric_valid0.0528.ckpt/"
]
TAR_ROOT = Path("/mnt/nvme2/share/chenyouyu/codespace/vggt_sz_ft/")

ZERO_TO_FP32 = "python ./python/zero_to_fp32.py {} {}"

for src_subtree in SRC_SUBTREE:
    # src_tree = SRC_ROOT + src_subtree
    # tar_tree = TAR_ROOT / src_subtree
    # tar_tree = tar_tree.parent
    # tar_tree.mkdir(parents=True, exist_ok=True)
    
    # print("downloading {} to {}".format(src_tree, str(tar_tree)))
    
    # cmd = CMD.format(str(tar_tree), src_tree)
    # _ = subprocess.call(cmd.split(' '))
    
    tar_tree = TAR_ROOT / src_subtree
    
    print("converting weights to {}...".format(str(tar_tree)))
    
    zero_to_fp32 = ZERO_TO_FP32.format(str(tar_tree), str(tar_tree))
    _ = subprocess.call(zero_to_fp32.split(' '))