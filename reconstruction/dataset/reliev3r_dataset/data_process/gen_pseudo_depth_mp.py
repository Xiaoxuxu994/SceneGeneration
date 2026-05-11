import os
# os.environ["NPU_VISIBLE_DEVICES"]="6"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="6"
import multiprocessing
import torch
from utils.platform import DEVICE_NAME, PLATFORM
if PLATFORM == "ASCEND":
    import torch_npu
    import torch_npu.npu as torch_device_lib
else:
    import torch.cuda as torch_device_lib
import json
import numpy as np
import cv2
import time

from glob import glob
from dotted_dict import DottedDict

from model.blocks.mono_depth_models import MonoDepthModel
from utils.image_utils import gray2hot

IMAGE_FOLDER_NAME = "images_4"
DEPTH_FOLDER_NAME = "moge2_depth"
TOTAL_PROCESS  = os.environ.get("C_TOTAL_PROCESS_COUNT", None)
PROCESS_OFFSET = os.environ.get("C_PROCESS_OFFSET", None)
NODE_PROCESS   = os.environ.get("C_NODE_PROCESS", torch_device_lib.device_count())
assert TOTAL_PROCESS is not None and PROCESS_OFFSET is not None
TOTAL_PROCESS, PROCESS_OFFSET, NODE_PROCESS = int(TOTAL_PROCESS), int(PROCESS_OFFSET), int(NODE_PROCESS)

# scene_list_json = os.path.join("/efs_data", "datasets", "DL3DV", "scene_list_train.json")
# root = "/efs_data/datasets/DL3DV"
root = "data"
scene_list_json = os.path.join(root, "scene_list_eval.json")
with open(scene_list_json, "r") as f:
    scene_list = json.load(f)

filtered_scene_list = scene_list
print("[ INFO ] {} scenes found to generate pseudo depth label.".format(len(filtered_scene_list)))

def run(rank):
    
    # torch_device_lib.set_device(rank)
    global_rank = rank + PROCESS_OFFSET
    
    # depth_model = MonoDepthModel(DottedDict({
    #     "name": "DepthAnythingV3", 
    #     "ckpt_path": {
    #         # "DepthAnythingV3": "./ckpt/depth-anything-v3", 
    #         # "DepthAnythingV3": "./ckpt/depth-anything-v3-mono-large", 
    #         "DepthAnythingV3": "./ckpt/depth-anything-v3-mono-metric-large", 
    #     }, 
    # })).to(device=DEVICE_NAME).eval()
    
    # depth_model = MonoDepthModel(DottedDict({
    #     "name": "DepthPro", 
    #     "ckpt_path": {
    #         "DepthPro": "./ckpt/depth-pro/depth_pro.pt", 
    #     }, 
    # })).to(device=DEVICE_NAME).eval()

    depth_model = MonoDepthModel(DottedDict({
        "name": "MoGe2", 
        "ckpt_path": {
            "MoGe2": "./ckpt/moge2/model.pt", 
        }, 
    })).to(device=DEVICE_NAME).eval()

    error_scenes = []

    total_time = 0.0
    total_samp = 0
    for idx, scene_folder in enumerate(filtered_scene_list):
        if idx % TOTAL_PROCESS != global_rank:
            continue
        # try:
        start_time = time.time()
        image_list = sorted(glob(os.path.join(scene_folder, IMAGE_FOLDER_NAME, "frame_*.png")))
        os.makedirs(os.path.join(scene_folder, DEPTH_FOLDER_NAME), exist_ok=True)
        depth_file_list = sorted(glob(os.path.join(scene_folder, DEPTH_FOLDER_NAME, "*.npy")))
        if len(image_list) == len(depth_file_list):
            continue
        for image_file in image_list:
            image = torch.from_numpy(np.array(cv2.imread(image_file))).to(device=DEVICE_NAME, dtype=torch.float32).flip(dims=(-1, ))
            image = image / 255
            image = image.permute(2, 0, 1)[None, None] # 1 1 1 h w
            mono_depth = depth_model.infer(image, f_px=torch.tensor(215.0, device=DEVICE_NAME))
            with open(os.path.join(scene_folder, DEPTH_FOLDER_NAME, image_file.split('/')[-1].split('.')[0] + ".npy"), "wb") as f:
                np.save(f, mono_depth[0].cpu().numpy()) # 1 h w
            # Visualize as .png
            # mono_depth = mono_depth.clamp(0, 1000)
            # mono_depth = torch.log10(mono_depth + 1)
            # mono_depth = gray2hot(mono_depth).squeeze().permute(1, 2, 0).flip(dims=(-1, )).cpu().numpy() * 255
            # cv2.imwrite(os.path.join(scene_folder, DEPTH_FOLDER_NAME, image_file.split('/')[-1]), mono_depth)
        end_time = time.time()
        total_time += end_time - start_time
        total_samp += 1
        print("[ INFO ] progress {}/{}, avg time per iter {:.2f} minites, {:.2f} hours left.".format(
            idx+1, len(filtered_scene_list), 
            total_time / total_samp / 60, 
            total_time / total_samp / 3600 * (len(filtered_scene_list) - idx - 1) / TOTAL_PROCESS))
        # except:
        #     error_scenes.append(scene_folder)
    
    with open(os.path.join(root, "globalrank{}_errorcase.json".format(global_rank)), "w+") as f:
        json.dump(error_scenes, f, indent="\t")

processes = []
for i in range(NODE_PROCESS):
    process = multiprocessing.Process(target=run, args=(i, ))
    processes.append(process)
    process.start()

for process in processes:
    process.join()
