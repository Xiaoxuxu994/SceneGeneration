import os
from glob import glob
import numpy as np
from tqdm import tqdm
from romav2 import RoMaV2
from argparse import ArgumentParser
import torch
import time

from ..utils.io import read_image_as_tensor
from utils.platform import DEVICE_NAME

BATCH_SIZE = 512
DS_RATE = 1

def main(args):
    clip_folders = sorted(glob(os.path.join(args.root, "sub_*", "*_*")))
    if args.proc_id is None:
        print("[ INFO ] {} clip folders found.".format(len(clip_folders)))
    else:
        clip_folders = clip_folders[args.proc_id::args.total_proc]
        print("[ INFO - PROC:{} ] {} clip folders found.".format(args.proc_id, len(clip_folders)))
    model = RoMaV2()
    total_time = 0.0
    for idx, clip_folder in enumerate(clip_folders):
        # if os.path.exists(os.path.join(clip_folder, "romav2")):
        #     continue
        start_time = time.time()
        image_files = sorted(glob(os.path.join(clip_folder, "rgb", "*.*")))
        clip_matching_infos = []
        image_tensor_list = [read_image_as_tensor(image_file).to(device=DEVICE_NAME) for image_file in image_files]
        image_tensor_list = torch.stack(image_tensor_list, dim=0)
        image_shape = image_tensor_list.shape[-2:]
        H_a, W_a = image_shape
        H_b, W_b = image_shape
        image_tensor_list = torch.nn.functional.interpolate(image_tensor_list, (224, 224), mode="bicubic")
        image_idx_pairs = []
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                image_idx_pairs.append((i, j))
        batched_idx_pairs = []
        for batch_start_idx in range(0, len(image_idx_pairs), BATCH_SIZE):
            batched_idx_pairs.append(image_idx_pairs[batch_start_idx: min(batch_start_idx + BATCH_SIZE, len(image_idx_pairs))])
        for batched_idx_pair in batched_idx_pairs:
            img_a = torch.cat([image_tensor_list[idx_pair[0]].unsqueeze(0) for idx_pair in batched_idx_pair], dim=0)
            img_b = torch.cat([image_tensor_list[idx_pair[1]].unsqueeze(0) for idx_pair in batched_idx_pair], dim=0)
            with torch.autocast(device_type=DEVICE_NAME, dtype=torch.bfloat16):
                preds = model(img_a, img_b)
            matches, overlaps = model.sample_batched(preds, 1000)
            kpts_a, kpts_b = model.to_pixel_coordinates(matches, H_a, W_a, H_b, W_b)
            for batch_idx in range(len(batched_idx_pair)):
                matching_info = {
                    "tracks": (kpts_a[batch_idx: batch_idx+1].cpu().numpy(), kpts_b[batch_idx: batch_idx+1].cpu().numpy()), 
                    "visibility": ((overlaps[batch_idx: batch_idx+1] > 0.5).cpu().numpy(), (overlaps[batch_idx: batch_idx+1] > 0.5).cpu().numpy()), 
                }
                clip_matching_infos.append(matching_info)
        os.makedirs(os.path.join(clip_folder, "romav2"), exist_ok=True)
        file_name = image_files[0].split("/")[-1].split(".")[0] + "_" + image_files[-1].split("/")[-1].split(".")[0] + ".npz"
        np.savez(os.path.join(clip_folder, "romav2", file_name), data=clip_matching_infos)
        end_time = time.time()
        total_time += (end_time - start_time) / 3600
        print("[ INFO - PROC:{} ] {}/{}, {:.2f}% processed, {:.2f} mins/clip, about {:.2f} hours left.".format(
            args.proc_id if args.proc_id is not None else 0, 
            idx+1, len(clip_folders), (idx+1)/len(clip_folders)*100, 
            total_time * 60 / (idx + 1), total_time / (idx + 1) * len(clip_folders) - total_time
        ))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--proc_id", type=int, default=None)
    parser.add_argument("--total_proc", type=int, default=None)
    args = parser.parse_args()
    main(args)