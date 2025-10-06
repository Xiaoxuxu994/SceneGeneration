# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from torchvision import transforms as TF

from mapanything.utils.image import preprocess_inputs
from mapanything.utils.inference import validate_input_views_for_inference, preprocess_input_views_for_inference


class Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]
        
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        

        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        file_name = scene.split("/")[-1]
                        scene_name = file_name.split(".")[0]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths


    def __len__(self):
        return len(self.all_scene_paths)


    def view_selector(self, frames):
        if len(frames) < self.config.training.num_views:
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views-2)
        image_indices = [start_frame, end_frame] + sampled_frames
        return image_indices

    def __getitem__(self, idx):
        # try:
        scene_path = self.all_scene_paths[idx].strip()
        data_json = json.load(open(scene_path, 'r'))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        if self.inference and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices= current_view_idx["context"] + current_view_idx["target"]
        else:
            # sample input and target views
            image_indices = self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
        frames_chosen = [frames[ic] for ic in image_indices]
        
        # Build MapAnything view dicts
        views = []
        world20 = torch.tensor(frames_chosen[0]["w2c"])  # world to cam0
        for cur_frame, image_path in zip(frames_chosen, image_paths_chosen):
            img = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)) # (H, W, 3) - [0, 255]
            
            fx, fy, cx, cy = cur_frame["fxfycxcy"]
            intrinsics_tensor = torch.tensor([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])
            
            pose_matrices = torch.linalg.inv(torch.tensor(cur_frame["w2c"]))  # current cam to world (4, 4) in OpenCV cam2world convention
            
            # Align coordinate system (OpenCV coordinate) to the first camera
            cam20 = world20 @ pose_matrices  # current cam to cam0
            
            view = {
                "img": img,
                "intrinsics": intrinsics_tensor,
                "camera_poses": cam20,
                "is_metric_scale": torch.tensor([False]), # (1,)
            }
        
            views.append(view)
        
        # Preprocess MapAnything view dict
        old_views = preprocess_inputs(views)
        old_views = validate_input_views_for_inference(old_views)
        old_views = preprocess_input_views_for_inference(old_views)
        views = []
        for view in old_views:
            view['img'] = view['img'].squeeze(0)
            view['camera_pose_quats'] = view['camera_pose_quats'].squeeze(0)
            view['camera_pose_trans'] = view['camera_pose_trans'].squeeze(0)
            view['ray_directions_cam'] = view['ray_directions_cam'].squeeze(0)
            view['is_metric_scale'] = view['is_metric_scale'].squeeze(0)
            view['data_norm_type'] = view['data_norm_type'][0]
            views.append(view)
        
        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        datum = {}
        datum["views"] = views
        datum["index"] = indices
        datum["scene_name"] = scene_name
        
        return datum
