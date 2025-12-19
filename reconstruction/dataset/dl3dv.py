import os
import sys
import numpy as np
import torch
import torch.nn as nn
import json
from random import shuffle
from glob import glob
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader


from .utils.camera_reader import read_camera_from_json, CustomCamera

from .utils import load_and_preprocess_images
from utils import PLATFORM, DEVICE_NAME
# MAX_IMG_SIZE = 518

TRAIN_IMAGE_FOLDER = "images_8"
EVAL_IMAGE_FOLDER  = "images_4"

class DL3DV_Dataset(Dataset):
    def __init__(self, cfg, data_type):
        assert data_type in ["train", "eval"]
        assert cfg.sample_views % 2 == 0
        assert cfg.sample_mode in ["stride"]
        self.root         = cfg.data_root # The path containing 1K, 2K, ..., 11K for DL3DV dataset
        self.input_num    = cfg.sample_views // 2
        self.output_num   = cfg.sample_views // 2
        self.data_type    = data_type
        self.samp_mode    = cfg.sample_mode
        self.samp_stride  = cfg.sample_segment // cfg.sample_views
        self.samp_segment = cfg.sample_segment
        self.reuse_scene  = cfg.reuse_scene
        self.load_preprocessed_matching = cfg.load_preprocessed_matching
        self.preprocessed_matching_path = cfg.preprocessed_matching_path
        if self.preprocessed_matching_path == "placeholder":
            self.preprocessed_matching_path = os.path.join(self.root, "matching_preprocess" + 
                ("_eval" if self.data_type == "eval" else ""))
        self.img_folder_name = TRAIN_IMAGE_FOLDER if self.data_type == "train" else EVAL_IMAGE_FOLDER
        
        if data_type == "train":
            self.scene_list = self.check_and_load_scenes(
                subsets=cfg.training_sets, 
                scene_file_name="scene_list_train.json", 
                image_folder_name=self.img_folder_name)
        else:
            self.scene_list = self.check_and_load_scenes(
                subsets=cfg.eval_sets, 
                scene_file_name="scene_list_eval.json", 
                image_folder_name=self.img_folder_name)
        
        self.create_meta_indices()
    
    def create_meta_indices(self):
        """Create meta index, which is the unit of training sample.
        """
        self.meta_indices = []
        for scene_idx, scene in enumerate(self.scene_list):
            # if scene_idx > 0:
            #     break
            scene_img_num = len(glob(os.path.join(scene, self.img_folder_name, "*.png")))
            if self.reuse_scene:
                for i in range(scene_img_num // self.samp_segment):
                    # if i * self.samp_segment != 0:
                    #     continue
                    self.meta_indices.append({
                        "scene_idx": scene_idx, 
                        "img_idx_range": (i * self.samp_segment, (i + 1) * self.samp_segment), 
                    })
            elif scene_img_num >= self.samp_segment:
                self.meta_indices.append({
                    "scene_idx": scene_idx, 
                    "img_idx_range": (0, self.samp_segment), 
                })
    
    def shuffle_meta_indices(self):
        shuffle(self.meta_indices)
    
    def check_and_load_scenes(self, subsets, scene_file_name, image_folder_name="images"):
        scene_list  = []
        for subset in subsets:
            subset_folder = os.path.join(self.root, subset, "*")
            subset_scenes = sorted(glob(subset_folder))
            scene_list.extend(subset_scenes)
        
        # Check file completeness
        checked_list = []
        if not os.path.exists(os.path.join(self.root, scene_file_name)):
            # If data completeness has not been checked, check the completeness and record the complete scenes. 
            for scene in tqdm(scene_list, total=len(scene_list), desc="Validating Data..."):
                if os.path.exists(os.path.join(scene, image_folder_name)): # and os.path.exists(os.path.join(scene, "sparse")):
                    image_names = glob(os.path.join(scene, image_folder_name, "*.png"))
                    if os.path.exists(os.path.join(scene, "transforms.json")):
                        try:
                            with open(os.path.join(scene, "transforms.json")) as f:
                                cam_info_nerf = json.load(f)
                            if len(image_names) == len(cam_info_nerf["frames"]) and len(image_names) >= self.input_num + self.output_num:
                                if len(image_names) > 100:
                                    max_val = 0
                                    for frame_info in cam_info_nerf["frames"]:
                                        trans_mat = np.array(frame_info["transform_matrix"])
                                        max_val = max(max_val, np.max(np.abs(trans_mat)))
                                    if max_val < 100:
                                        checked_list.append(scene)
                        except:
                            print("[ Warning ] Found error in scene {}".format(scene))
            with open(os.path.join(self.root, scene_file_name), "w+") as f:
                json.dump(checked_list, f, indent='\t')
            print("[ INFO ] Found {} scene validated.".format(len(checked_list)))
        else:
            # Data completeness has already been checked, directly load from the record. 
            print("[ INFO ] Pre-checked scene list found.")
            with open(os.path.join(self.root, scene_file_name)) as f:
                checked_list = json.load(f)
            print("[ INFO ] Found {} scene validated.".format(len(checked_list)))
        
        return checked_list

    def __len__(self):
        return len(self.meta_indices)
    
    def __getitem__(self, idx):
        
        meta_index = self.meta_indices[idx]
        scene_idx  = meta_index["scene_idx"]
        img_idx_range = meta_index["img_idx_range"]
        
        scene = self.scene_list[scene_idx]
        
        image_names = sorted(glob(os.path.join(scene, self.img_folder_name, "*.png")))
        depth_names = sorted(glob(os.path.join(scene, "moge2_depth", "*.npy")))
        
        assert len(image_names) >= self.input_num + self.output_num
        if self.samp_mode == "stride":
            # Sample input views from beginning with a given stride.
            # assert self.output_num == 0, "This mode has only been developed for evaluation yet."
            assert self.samp_stride >= 1, "Input view sample stride should be not less than 1."
            indices = np.arange(len(image_names))[img_idx_range[0]: img_idx_range[1]][::self.samp_stride]
            if len(indices) > self.input_num + self.output_num:
                indices = indices[:self.input_num + self.output_num]
            assert len(indices) == self.input_num + self.output_num
            # input_idx  = indices[0::2]
            # output_idx = indices[1::2]
            input_idx  = indices
            output_idx = []
        else:
            raise NotImplementedError("Sample mode {} is not implemented.".format(self.samp_mode))
        
        # Read images
        input_image_names = [image_names[idx] for idx in input_idx]
        input_depth_names = [depth_names[idx] for idx in input_idx] if len(depth_names) == len(image_names) else None
        try:
            input_images, input_depths, _, raw_img_shape = load_and_preprocess_images(input_image_names, input_depth_names, device="cpu")
        except:
            return None
        if input_depths is not None:
            input_depths = input_depths.clamp(min=1e-3, max=1000)
        
        # Read cameras
        h, w = input_images.shape[-2:]
        with open(os.path.join(scene, "transforms.json"), "r+") as fp:
            transform_dict = json.load(fp)
        cam_extrinsics, cam_intrinsics = read_camera_from_json(transform_dict, h, w) # extrinsics here is opencv convention, world to camera.
        cam_extrinsics = torch.tensor(cam_extrinsics).to(torch.float32)
        cam_intrinsics = torch.tensor(cam_intrinsics).to(torch.float32)

        cam_infos = [CustomCamera(extr, intr, *input_images.shape[-2:]) for extr, intr in zip(cam_extrinsics, cam_intrinsics)]
        input_cams  = [cam_infos[idx] for idx in input_idx]
        input_viewmats  = torch.stack([cam_extrinsics[idx] for idx in input_idx]) # v 3 3
        input_t = input_viewmats[..., :3, 3] # v 3
        input_t = input_t / (input_t.norm(dim=-1).mean(dim=-1)[None, None] + 1e-8) # v 3
        input_viewmats[..., :3, 3] = input_t
        
        input_Ks  = torch.stack([cam_intrinsics[idx] for idx in input_idx])
        input_Ks[..., 0, :] *= w
        input_Ks[..., 1, :] *= h
        
        matching_infos = None
        matching_infos_file = os.path.join(self.preprocessed_matching_path, "{:09d}.npy".format(idx))
        if self.load_preprocessed_matching:
            if os.path.exists(matching_infos_file):
                matching_infos = np.load(matching_infos_file, allow_pickle=True).item()
                for key in matching_infos.keys():
                    matching_infos[key]["matches"] = torch.from_numpy(matching_infos[key]["matches"])
                    matching_infos[key]["visibilities"] = torch.from_numpy(matching_infos[key]["visibilities"])
            else:
                return None
        
        return {
            "scene_names":       scene.split('/')[-1], 
            "input_images":      input_images, 
            "input_depths":      input_depths if input_depths is not None else None, 
            "sample_indices":    indices, 
            "input_viewmats":    input_viewmats, # world to camera
            "input_Ks":          input_Ks, 
            "input_image_names": input_image_names, 
            "data_idxs":         idx, 
            "matching_infos":    matching_infos, 
        }

def DL3DV_collate_fn(batch_input):
    if batch_input[0] is None:
        return None
    scene_names    = []
    input_images   = []
    input_depths   = []
    sample_indices = []
    input_viewmats = []
    input_Ks       = []
    data_idxs      = []
    for data in batch_input:
        scene_names.append(data["scene_names"])
        input_images.append(data["input_images"])
        input_depths.append(data["input_depths"])
        sample_indices.append(data["sample_indices"])
        input_viewmats.append(data["input_viewmats"])
        input_Ks.append(data["input_Ks"])
        data_idxs.extend(data["data_idxs"])
    input_images   = torch.stack(input_images)
    input_depths   = torch.stack(input_depths) if input_depths[0] is not None else None
    input_viewmats = torch.stack(input_viewmats)
    input_Ks       = torch.stack(input_Ks)
    return {
        "scene_names":    scene_names,
        "input_images":   input_images, 
        "input_depths":   input_depths, 
        "sample_indices": sample_indices, 
        "input_viewmats": input_viewmats, 
        "input_Ks":       input_Ks, 
        "matching_infos": batch_input[0]["matching_infos"], 
        "data_idxs":      data_idxs, 
    }

