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

from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.image import find_closest_aspect_ratio


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


    def preprocess_frames(
        self,
        frames_chosen,
        image_paths_chosen,
        resize_mode="fixed_mapping",
        norm_type="identity",
        size=None,
        patch_size=14,
        resolution_set=518,
        verbose=False,
    ):
        """
        MapAnything image preprocessing with corresponding intrinsics transformations
        """
        # Validate resize_mode and size parameter requirements
        valid_resize_modes = ["fixed_mapping", "longest_side", "square", "fixed_size"]
        if resize_mode not in valid_resize_modes:
            raise ValueError(
                f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
            )

        if resize_mode in ["longest_side", "square", "fixed_size"] and size is None:
            raise ValueError(f"Size parameter is required for resize_mode='{resize_mode}'")

        # Validate size type based on resize mode
        if resize_mode in ["longest_side", "square"]:
            if not isinstance(size, int):
                raise ValueError(
                    f"Size must be an int for resize_mode='{resize_mode}', got {type(size)}"
                )
        elif resize_mode == "fixed_size":
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise ValueError(
                    f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
                )
            if not all(isinstance(x, int) for x in size):
                raise ValueError(
                    f"Size values must be integers for resize_mode='fixed_size', got {size}"
                )
        
        ## First pass: Extract all images and collect aspect ratios
        aspect_ratios = []
        for cur_frame, image_path in zip(frames_chosen, image_paths_chosen):
            # Open image
            img = Image.open(image_path)

            # If there's an alpha channel, blend onto white background:
            if img.mode == "RGBA":
                # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Alpha composite onto the white background
                img = Image.alpha_composite(background, img)

            # Now convert to "RGB" (this step assigns white for transparent areas)
            img = img.convert("RGB")

            width, height = img.size
            aspect_ratios.append(width / height)
        
        # Check for empty list
        if not aspect_ratios:
            raise ValueError("At least 1 image is required")
        
        # Calculate average aspect ratio and determine target size
        average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
        if verbose:
            print(
                f"Calculated average aspect ratio: {average_aspect_ratio:.3f} from {len(aspect_ratios)} images"
            )

        # Determine target size for all images based on resize mode
        if resize_mode == "fixed_mapping":
            # Resolution mappings are already compatible with their respective patch sizes
            target_width, target_height = find_closest_aspect_ratio(
                average_aspect_ratio, resolution_set
            )
            target_size = (target_width, target_height)
        elif resize_mode == "square":
            target_size = (
                round((size // patch_size)) * patch_size,
                round((size // patch_size)) * patch_size,
            )
        elif resize_mode == "longest_side":
            # Use average aspect ratio to determine size for all images
            if average_aspect_ratio >= 1:  # Landscape or square
                target_size = (
                    size,
                    round((size // patch_size) / average_aspect_ratio) * patch_size,
                )
            else:  # Portrait
                target_size = (
                    round((size // patch_size) * average_aspect_ratio) * patch_size,
                    size,
                )
        elif resize_mode == "fixed_size":
            # Use exact size provided, aligned to patch_size
            target_size = (
                (size[0] // patch_size) * patch_size,
                (size[1] // patch_size) * patch_size,
            )

        if verbose:
            print(
                f"Using target resolution {target_size[0]}x{target_size[1]} (W x H) for all views"
            )

        # Get the image normalization function based on the norm_type
        if norm_type in IMAGE_NORMALIZATION_DICT.keys():
            img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
            ImgNorm = TF.Compose(
                [TF.ToTensor(), TF.Normalize(mean=img_norm.mean, std=img_norm.std)]
            )
        else:
            raise ValueError(
                f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
            )
                    
        ## Second pass: Resize all images and multi-modal inputs
        images = []
        intrinsics = []
        c2ws = []
        
        for cur_frame, image_path in zip(frames_chosen, image_paths_chosen):
            # Open image
            img = Image.open(image_path)

            # If there's an alpha channel, blend onto white background:
            if img.mode == "RGBA":
                # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Alpha composite onto the white background
                img = Image.alpha_composite(background, img)

            # Now convert to "RGB" (this step assigns white for transparent areas)
            img = img.convert("RGB")

            # Get intrinsics
            fx, fy, cx, cy = cur_frame["fxfycxcy"]
            intrinsics_matrix =  np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])
            
            # Process all inputs with a single call to crop_resize_if_necessary
            img, intrinsics_matrix = crop_resize_if_necessary(
                image=img,
                resolution=target_size,
                intrinsics=intrinsics_matrix,
            )
            
            # Normalize image
            img = ImgNorm(img)
            
            # Get extrinsics in OpenCV cam2world convention
            c2w = torch.linalg.inv(torch.tensor(cur_frame["w2c"]))
            
            # Convert intrinsics to LVSM format
            fxfycxcy = torch.tensor([intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[2, 0], intrinsics_matrix[2, 1]]).float()
            
            images.append(img)
            intrinsics.append(fxfycxcy)
            c2ws.append(c2w)
        
        images = torch.stack(images)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
                
        return images, intrinsics, c2ws


    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the first camera
        in_02w = in_c2ws[0, ...]  # cam0 to world
        in_w20 = torch.linalg.inv(in_02w)  # world to cam0
        in_c2ws = in_w20 @ in_c2ws  # all cams to cam0
        
        # # Rescale the whole scene to a fixed scale
        # scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        # scene_scale = scene_scale_factor * scene_scale
        # in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws


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
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

        # centerize and scale the poses (for unbounded scenes)
        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        is_metric_scale = torch.tensor([False]*len(image_indices)).unsqueeze(-1)  # [v, 1]  # TODO: remove hard-coded

        datum = {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene_name,
            "is_metric_scale": is_metric_scale,
            "norm_type": 'identity',  # TODO: remove hard-coded
        }
                        
        return datum
