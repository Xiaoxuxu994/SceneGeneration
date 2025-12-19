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

from .utils import load_and_preprocess_images_from_meta
from utils import PLATFORM, DEVICE_NAME
from utils.log_utils import depth2pointmap
from utils.common import calc_extr_inv
# MAX_IMG_SIZE = 518

TRAIN_IMAGE_FOLDER = "images_8"
EVAL_IMAGE_FOLDER  = "images_4"

class DL3DV_WAI_Dataset(Dataset):
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
        
        with open(cfg["{}_data_list_file".format(data_type)], "r+") as fp:
            self.scene_list = json.load(fp)
        
        self.create_meta_indices()
    
    def create_meta_indices(self):
        """Create meta index, which is the unit of training sample.
        """
        self.meta_indices = []
        for scene_idx, scene in enumerate(self.scene_list):
            scene = os.path.join(self.root, scene)
            with open(os.path.join(scene, "scene_meta.json"), "r+") as fp:
                scene_meta = json.load(fp)
            scene_img_num = len(scene_meta["frames"])
            if self.reuse_scene:
                for i in range(scene_img_num // self.samp_segment):
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

    def __len__(self):
        return len(self.meta_indices)
    
    def __getitem__(self, idx):
        
        meta_index = self.meta_indices[idx]
        scene_idx  = meta_index["scene_idx"]
        img_idx_range = meta_index["img_idx_range"]
        
        scene = self.scene_list[scene_idx]
        scene = os.path.join(self.root, scene)
        with open(os.path.join(self.root, scene, "scene_meta.json"), "r+") as fp:
            scene_meta = json.load(fp)
        
        frame_info_names = [dict(image=os.path.join(scene, frame_meta["image"]), 
                                 mvsa_depth=os.path.join(scene, frame_meta["mvsanywhere_depth"]), 
                                 mvsa_depth_conf=os.path.join(scene, frame_meta["mvsanywhere_depth_confidence"]), 
                                #  moge2_mask=os.path.join(scene, frame_meta["moge2_mask"]), 
                                 ) for frame_meta in scene_meta["frames"]]
        frame_info_names = sorted(frame_info_names, key=lambda x: x["image"])
        
        assert len(frame_info_names) >= self.input_num + self.output_num
        if self.samp_mode == "stride":
            # Sample input views from beginning with a given stride.
            # assert self.output_num == 0, "This mode has only been developed for evaluation yet."
            assert self.samp_stride >= 1, "Input view sample stride should be not less than 1."
            indices = np.arange(len(frame_info_names))[img_idx_range[0]: img_idx_range[1]][::self.samp_stride]
            if len(indices) > self.input_num + self.output_num:
                indices = indices[:self.input_num + self.output_num]
            assert len(indices) == self.input_num + self.output_num
            input_idx = indices
        else:
            raise NotImplementedError("Sample mode {} is not implemented.".format(self.samp_mode))
        
        frame_info_names = [frame_info_names[idx] for idx in input_idx]
        
        # Read images
        try:
            input_images, input_depths, input_masks, raw_img_shape = load_and_preprocess_images_from_meta(
                frame_metas=frame_info_names, device="cpu")
        except:
            return None
                
        # Read cameras
        h, w = input_images.shape[-2:]
        cam_extrinsics, cam_intrinsics = read_camera_from_json(scene_meta, h, w, apply_trans=False) # extrinsics here is opencv convention, world to camera.
        cam_extrinsics = torch.tensor(cam_extrinsics).to(torch.float32)
        cam_intrinsics = torch.tensor(cam_intrinsics).to(torch.float32)

        input_viewmats = torch.stack([cam_extrinsics[idx] for idx in input_idx]) # v 4 4, w2c
        
        ### Transform all input view mats from world-to-camera to first-to-others
        # input_viewmats = calc_extr_inv(input_viewmats[:1]).expand_as(input_viewmats) @ input_viewmats # v 4 4, first to others
        # input_viewmats = input_viewmats @ calc_extr_inv(input_viewmats[:1]).expand_as(input_viewmats) # v 4 4, first to others
        
        input_Ks = torch.stack([cam_intrinsics[idx] for idx in input_idx])
        input_Ks[..., 0, :] *= w
        input_Ks[..., 1, :] *= h
        input_points = depth2pointmap(depths=input_depths, extrs=input_viewmats, intrs=input_Ks, device="cpu") # V H W 3
        
        ### Normalize the point maps, depths and view mats following VGGT. 
        # norm_factor = input_points.norm(p=2, dim=-1).mean() + 1e-8 # scalar
        # input_points = input_points / norm_factor
        # input_depths = input_depths / norm_factor
        # input_viewmats[..., :3, 3] = input_viewmats[..., :3, 3] / norm_factor
        
        return {
            "scene_names":    scene.split('/')[-1], 
            "input_images":   input_images, 
            "input_depths":   input_depths, 
            "input_points":   input_points, 
            "input_masks":    input_masks, 
            "sample_indices": indices, 
            "input_viewmats": input_viewmats, # world-to-camera, first frame as the world
            "input_Ks":       input_Ks, 
            "data_idx":       idx, 
        }

def DL3DV_WAI_collate_fn(batch_input):
    if batch_input[0] is None:
        return None
    collate_dict = {}
    for key in batch_input[0].keys():
        if isinstance(batch_input[0][key], torch.Tensor):
            inputs = [batch_input[idx][key] for idx in range(len(batch_input))]
            inputs = torch.stack(inputs)
            collate_dict.update({key: inputs})
        elif key == "matching_infos":
            collate_dict.update({key: batch_input[0]["matching_infos"]})
        else:
            inputs = [batch_input[idx][key] for idx in range(len(batch_input))]
            collate_dict.update({key: inputs})
    return collate_dict

