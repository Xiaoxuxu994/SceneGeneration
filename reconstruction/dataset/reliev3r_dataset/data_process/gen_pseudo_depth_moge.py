import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from glob import glob
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import time
from dotted_dict import DottedDict
import cv2

from ..utils.io import read_image_as_tensor
from utils.platform import DEVICE_NAME
from model.blocks.mono_depth_models import MonoDepthModel

IMAGE_FOLDER_NAME = "rgb"
DEPTH_FOLDER_NAME = "moge_depth"
DEPTH_MASK_FOLDER_NAME = "moge_depth_mask"


def main(args):
    clip_folders = sorted(glob(os.path.join(args.root, "sub_*", "*_*")))
    if args.proc_id is None:
        print("[ INFO ] {} clip folders found.".format(len(clip_folders)))
    else:
        clip_folders = clip_folders[args.proc_id::args.total_proc]
        print("[ INFO - PROC:{} ] {} clip folders found.".format(args.proc_id, len(clip_folders)))
    
    depth_model = MonoDepthModel(DottedDict({
        "name": "MoGe2", 
        "ckpt_path": {
            "MoGe2": "./ckpt/moge2/model.pt", 
        }, 
    })).to(device=DEVICE_NAME).eval()
    print("[ INFO ] Depth model initialized.")
    
    total_time = 0.0
    processed_clip_count = 0
    for clip_idx, clip_folder in enumerate(clip_folders):
        clip_image_list = sorted(glob(os.path.join(clip_folder, IMAGE_FOLDER_NAME, "*")))
        clip_depth_list = sorted(glob(os.path.join(clip_folder, DEPTH_FOLDER_NAME, "*")))
        if len(clip_image_list) == len(clip_depth_list):
            continue
        depth_output_folder = os.path.join(clip_folder, DEPTH_FOLDER_NAME)
        depth_mask_output_folder = os.path.join(clip_folder, DEPTH_MASK_FOLDER_NAME)
        os.makedirs(depth_output_folder, exist_ok=True)
        os.makedirs(depth_mask_output_folder, exist_ok=True)
        start_time = time.time()
        intrinsics = []
        for image_file in clip_image_list:
            image_base_name = os.path.basename(image_file).split(".")[0]
            image = read_image_as_tensor(image_file).to(device=DEVICE_NAME) # 3 h w
            image = image.unsqueeze(0).unsqueeze(0) # 1 1 3 h w
            mono_depth, mono_depth_mask, intrinsic = depth_model.infer(image)
            mono_depth = mono_depth[0].permute(1, 2, 0)
            mono_depth = mono_depth * 100
            mono_depth[mono_depth.isinf()] = 65535
            mono_depth[mono_depth < 1] = 1
            mono_depth[mono_depth > 65535] = 65535
            mono_depth = mono_depth.to(torch.uint16)
            mono_depth_mask = mono_depth_mask[0].permute(1, 2, 0)
            mono_depth = mono_depth.cpu().numpy()
            mono_depth_mask = (mono_depth_mask.cpu().to(dtype=torch.uint8) * 255).numpy()
            cv2.imwrite(os.path.join(depth_output_folder, image_base_name + ".png"), mono_depth)
            cv2.imwrite(os.path.join(depth_mask_output_folder, image_base_name + ".png"), mono_depth_mask)
            intrinsics.append(intrinsic)
        if intrinsics[0] is not None and args.store_intr:
            camera_file_name = os.path.join(clip_folder, clip_folder.split('/')[-1] + ".npy")
            intrinsics = torch.cat(intrinsics, dim=0)
            intrinsics = intrinsics.mean(dim=0, keepdim=True)
            intrinsics = torch.cat([intrinsics for _ in range(len(clip_image_list))], dim=0)
            camera_data = dict(extr_mat=None, intr_mat=intrinsics.cpu().numpy())
            np.save(camera_file_name, camera_data, allow_pickle=True)
        end_time = time.time()
        total_time += end_time - start_time
        processed_clip_count += 1
        print("[ INFO - PROC:{} ] clip {}/{}, {:.2f}% processed, {:.2f} mins/clip, about {:.2f} hours left.".format(
            args.proc_id, 
            clip_idx + 1, len(clip_folders), (clip_idx + 1) / len(clip_folders) * 100, 
            total_time / 60 / (processed_clip_count + 1), 
            total_time / processed_clip_count / 3600 * (len(clip_folders) - (clip_idx + 1))
        ))
    print("[ INFO - PROC:{} ] Complete.".format(args.proc_id))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--proc_id", type=int, default=None)
    parser.add_argument("--total_proc", type=int, default=None)
    parser.add_argument("--store_intr", action="store_true")
    args = parser.parse_args()
    main(args)