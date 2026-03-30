# benchmark/datasets/ase.py
from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import imageio
from PIL import Image
from torchvision import transforms as TF
import cv2

# from scipy.sparse import csr_array
# from scipy.sparse.csgraph import minimum_spanning_tree

# from vggt.utils.load_fn import load_and_preprocess_images
# from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch

from lib.utils import rdepth_to_zdepth, K_denormalize
from lib.matching import *

class ASEDataset(BaseDataset):
    """
    ASE:
      NRGBD/<seq>/
        undistort_rgb/vignette{id}.jpg # id is zero-padded 7 digits
        undistort_depth/depth{id}.png  # id is zero-padded 7 digits
        trajectory.csv                 # each line is a 3-dim T & 4-dim quaterion
        
    or simply input the sample sequence list:
        [
            {
                "input_images": <path>,    # read this
                "input_depths": <path>,    # read this
                "input_depth_masks": None,
                "input_extrs": np.array,
                "input_intrs": np.array,
            }
        ]
    """
    
    DEPTH_SCALE = 1000.0  # mm -> m
    NUM_TEST = 50
    RGB_WIDTH = 518
    RGB_HEIGHT = 518
    CROP_BORDER = 28
    
    def __init__(self, root: str | Path, num_seq: int = NUM_TEST, depth_scale: float = DEPTH_SCALE):
        super().__init__(root, num_seq)
        
        self.depth_scale = depth_scale
        
        if root.endswith(".npz"):
            self.sample_sequences = np.load(root, allow_pickle=True)
        else:
            raise NotImplementedError
        
    def list_sequences(self, split):
        return self.sample_sequences.files[-self.num_seq:]
    
    def count_seq_len(self, seq):
        sequence = self.sample_sequences[seq].item()
        rgbs = sequence["input_images"]
        depths = sequence["input_depths"]
        poses = sequence["input_extrs"]
        
        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)
    
    def make_paths(self, seq, frame_ids):
        sequence = self.sample_sequences[seq].item()
        images = [sequence["input_images"][i].replace("efs-mi-east4-3", "/efs-m3ga") for i in frame_ids]
        depths = [sequence["input_depths"][i].replace("efs-mi-east4-3", "/efs-m3ga") for i in frame_ids]
        
        return images, depths
    
    def _load_rgb(self, image_paths: List[str], 
            target_width=518, target_height=518, border=28):
        """default RGBs in ASE have black corners
            cropping input ASE RGBs from 518x518 to 462x462
        """
        
        # crop input images
        image_pils, images_cpu = [], []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            
            # resize to 518x518
            img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
            
            # crop to 462x462
            left, top, right, bottom = int(border), int(border), int(target_width-border), int(target_height-border)
            img = img.crop((left, top, right, bottom))        
    
            image_pils.append(img)
            img_tensor = TF.ToTensor()(img)
            images_cpu.append(img_tensor)
            
        images_cpu = torch.stack(images_cpu)
        
        return image_pils, images_cpu
    
    def _load_depth(self, depth_paths: List[str], K_norm: torch.Tensor, 
            target_width=518, target_height=518, border=28) -> torch.Tensor:
        """default RGBs in ASE have black corners
            cropping GT depths to match RGBs from 518x518 to 462x462
        """
        
        depths = []
        for depth_path in depth_paths:
            # loading with PIL has value issues
            depth = imageio.v2.imread(depth_path).astype(np.float32)
            
            # resize to 518x518
            depth = cv2.resize(depth, (target_width, target_height), interpolation = cv2.INTER_NEAREST)
            
            # to tensor
            depth = np.array(depth).astype(np.float32)
            depths.append(depth)
            
        depths = np.stack(depths)
        depths = depths / self.depth_scale  # mm -> m
        depths = torch.from_numpy(depths)
        depths = rdepth_to_zdepth(depths.unsqueeze(1), K_norm).squeeze(1) # 518x518
        
        # crop to 462x462
        left, top, right, bottom = int(border), int(border), int(target_width-border), int(target_height-border)
        depths = depths[:, left:right, top:bottom]
            
        return depths # (V,462,462) meters
    
    def _load_w2c(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        sequence = self.sample_sequences[seq].item()
        poses = np.stack([sequence["input_extrs"][i] for i in frame_ids])
        
        return torch.from_numpy(poses)
    
    def _load_K(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        sequence = self.sample_sequences[seq].item()
        K = np.stack([sequence["input_intrs"][i] for i in frame_ids])
    
        return torch.from_numpy(K).float()
    
    """reimplement the method due to image cropping"""
    def _denormalize_K(self, norm_K, W, H, crop_left=28, crop_top=28):
        pix_K = norm_K.clone()
        pix_K[..., 0, 0] = norm_K[..., 0, 0] * W
        pix_K[..., 1, 1] = norm_K[..., 1, 1] * H
        pix_K[..., 0, 2] = norm_K[..., 0, 2] * W - crop_left
        pix_K[..., 1, 2] = norm_K[..., 1, 2] * H - crop_top
        pix_K[..., 2, 2] = 1.0
        return pix_K
    
    # def _normalize_K(self, pix_K, H, W):
    #     norm_K = pix_K.clone()
    #     norm_K[..., 0, 0] = pix_K[..., 0, 0] / W
    #     norm_K[..., 1, 1] = pix_K[..., 1, 1] / H
    #     norm_K[..., 0, 2] = pix_K[..., 0, 2] / W
    #     norm_K[..., 1, 2] = pix_K[..., 1, 2] / H
    #     norm_K[..., 2, 2] = 1.0
    #     return norm_K
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, depth_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths) # (V,462,462)
        raw_w2c = self._load_w2c(seq, frame_ids)             # (V,4,4)
        raw_K = self._load_K(seq, frame_ids)                 # (V,3,3), normalized
        depths = self._load_depth(depth_paths, raw_K)        # (V,462,462) meters, no resize
        
        # _, H, W = depths.shape
        # raw_K = K_denormalize(raw_K, W, H)
        raw_K = self._denormalize_K(raw_K, self.RGB_WIDTH, self.RGB_HEIGHT, self.CROP_BORDER, self.CROP_BORDER)
        
        # from lib.geometry import depth_to_cam_coords_points
        # local_points = depth_to_cam_coords_points(depths.numpy()[0], raw_K.numpy()[0])
        
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pts = local_points.reshape(-1, 3)
        # pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        # c = images_cpu.numpy()[0].transpose((1, 2, 0)).reshape(-1, 3)
        # pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float32))
        # o3d.io.write_point_cloud("temp.ply", pcd)
        # exit()
        
        return SequenceBatch(
            seq=seq,
            frame_ids=frame_ids,
            image_paths=image_paths,
            pose_paths=image_paths,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            raw_w2c=raw_w2c,
            raw_K=raw_K,
            meta={
                "dataset": "ase",
                "DEPTH_SCALE": self.DEPTH_SCALE,
            },
        )
    
    # def view_sampler_iterative(self, visibs: torch.Tensor, max_seq_len: int, seq_len: int, thre: float):
    #     """
    #     Sample views of a sample from a clip in an iterative approach.
    #     Given a sampled view set S from a clip C, each time add a view x in C-S, which has a minimal maximum covisibility with views in S that \
    #         above a given threshold. In this way, the given threshold represents sparcity of the sampled views.

    #     Args:
    #         visibs (torch.Tensor): Covisibility of pair-wise tracking pixels. Tensor of shape (V(V-1)/2, gs*gs)
    #         clip_viwe_num (int): Number of views in the clip.
    #     """
    #     visib_mat = deflat_visibility_mat(visibs, max_seq_len) # v v, upper-triangle matrix.
    #     visib_mat[visib_mat < thre] = 0
        
    #     view_ids = []
    #     sampled_mask = torch.zeros(max_seq_len, dtype=torch.bool)
    #     # first_view_id = legal_roots[np.random.randint(low=0, high=len(legal_roots))]
    #     first_view_id = np.random.randint(low=0, high=max_seq_len)
    #     view_ids.append(first_view_id)
    #     sampled_mask[first_view_id] = True
    #     sampled_view_cnt = 1
    #     while sampled_view_cnt < seq_len:
    #         _visib_mat = visib_mat.clone()
    #         _visib_mat[sampled_mask] = 0
    #         _visib_mat[:, ~sampled_mask] = 0
    #         _visib_vec = _visib_mat.max(dim=-1).values # v
    #         _visib_vec[_visib_vec == 0] = 2
    #         new_view_id = _visib_vec.min(dim=0).indices
    #         if _visib_vec[new_view_id] == 2:
    #             return None
    #         else:
    #             view_ids.append(new_view_id.item())
    #             sampled_mask[new_view_id] = True
    #             sampled_view_cnt += 1
                
    #     return np.array(view_ids)
        
    # def get_frame_ids(
    #     self,
    #     seq: str,
    #     seq_len: int,                  # actual length of the sampled views from the original clip
    #     out_dir: str | Path,
    #     *,
    #     mode: str = "stride",          # "stride" | "random"
    #     stride: int = 1,
    #     max_seq_len: int = -1,         # maximum length of the original clip, default to use all
    #     seed: int = 0,
    #     reuse: bool = True,
    #     fname: str = "frame_ids.npy",
    #     max_retry: int = 10
    # ) -> List[int]:
    #     out_dir = Path(out_dir)
    #     frame_id_dir = out_dir / "frame_ids" / seq
    #     frame_id_dir.mkdir(parents=True, exist_ok=True)
    #     fp = frame_id_dir / fname

    #     if reuse and fp.exists():
    #         ids = np.load(fp).astype(np.int64).tolist()
    #         print(f"[INFO] Load frame id from {fp}")
    #     else:
    #         mode = str(mode).lower()

    #         assert seq_len > 0, "length of the sampled sequence must be greater than 0"

    #         if mode == "stride":
    #             stride = max(1, int(stride))
    #             actual_seq_len = self.count_seq_len(seq)
    #             max_seq_len = actual_seq_len if max_seq_len == -1 else min(actual_seq_len, max_seq_len)
            
    #             ids = list(range(0, max_seq_len, stride))
    #             ids = ids[:seq_len]

    #         elif mode == "random":
    #             # np.random.seed(int(seed))
                
    #             actual_seq_len = self.count_seq_len(seq)
    #             max_seq_len = actual_seq_len if max_seq_len == -1 else min(actual_seq_len, max_seq_len)
    #             all_ids = [int(i) for i in range(max_seq_len)]
                
    #             batch = self.load_sequence(seq, all_ids)
                        
    #             depths          = batch.depths           # V,H,W
    #             w2c             = batch.raw_w2c.float()  # V,4,4
    #             raw_K           = batch.raw_K            # V,3,3
    #             grid_size       = 100
    #             match_eps       = 0.01
    #             samples_to_keep = 10000
    #             min_thres       = 0.08
                
    #             V,H,W = depths.shape
    #             norm_K = self._normalize_K(raw_K, H, W)
                
    #             ## Dataset extrinsics are defined as world-to-camera. Transform to camera-to-world when passed.
    #             _, _, visibs = exhaustive_geo2match(
    #                 depths, closed_form_inverse_se3(w2c), norm_K, eps=match_eps, samples_to_keep=samples_to_keep, grid_size=grid_size)
    #             visibs = visibs.to(dtype=torch.float32).mean(dim=-1)
                
    #             for try_id in range(max_retry):
    #                 ids = self.view_sampler_iterative(visibs, max_seq_len, seq_len, min_thres)
                    
    #                 if ids is None:
    #                     ids = np.zeros_like(ids)
    #                     print(f"[INFO] not a valid sequence - starting sampling trial {try_id+1}")
    #                     continue
                            
    #         else:
    #             raise ValueError(f"Unknown mode={mode}, expected stride|random")
            
    #         ids.sort()
    #         ids = ids.tolist()
    #         np.save(fp, np.array(ids, dtype=np.int32))
        
    #     return ids