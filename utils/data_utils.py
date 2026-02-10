import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
import imageio


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)



class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def compute_rays_in_camera_frame(self, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Compute ray origins and directions in the camera coordinate frame.
        Rays originate at the camera center (0,0,0) and point into +z.

        Args:
            fxfycxcy (torch.tensor): [b, v, 4] intrinsics (fx, fy, cx, cy)
            h (int, optional): target image height. If None, estimated from cy.
            w (int, optional): target image width. If None, estimated from cx.
            device (str): device for torch tensors.

        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w] ray origins
            ray_d (torch.tensor): [b, v, 3, h, w] ray directions (normalized)
        """
        b, v = fxfycxcy.size()[:2]

        fx, fy, cx, cy = (
            fxfycxcy[:, :, 0],
            fxfycxcy[:, :, 1],
            fxfycxcy[:, :, 2],
            fxfycxcy[:, :, 3],
        )

        # Original resolution from intrinsics
        h_orig = int(2 * cy.max().item())
        w_orig = int(2 * cx.max().item())
        if h is None or w is None:
            h, w = h_orig, w_orig

        # Scale intrinsics if target resolution differs
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)

        # Pixel grid
        y, x = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)

        # Project to normalized camera coordinates (pixel centers with +0.5 offset)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)

        # Ray directions (in camera frame, no c2w applied)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # normalize

        # Ray origins (camera center in its own frame)
        ray_o = torch.zeros_like(ray_d)
        
        # Reshape back to [b, v, 3, h, w]
        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d
    
    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True):
        """
        Splits the input data batch into input and target sets.
        
        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            target_has_input (bool): If True, target includes input views.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if (self.config.training.get("dynamic_input_view_num", False) 
            and (not self.config.inference.get("if_inference", False))):
            self.config.training.num_input_views = np.random.choice([2, 4, 8, 16, 32])
        

        input_dict, target_dict = {}, {}
        # index = [] save for future use if we want to select specific views

        num_target_views, num_views, bs = self.config.training.num_target_views, data_batch["c2w"].size(1), data_batch["image"].size(0)
        assert num_target_views < num_views, f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."
        
        # Special case of ScanNet++: v_input = v_target and non-constant num_views
        if (self.config.training.num_target_views == self.config.training.num_input_views) \
            & (data_batch["c2w"].size(1) != self.config.training.num_views):
                self.config.training.num_views = min(self.config.training.num_views, data_batch["c2w"].size(1))
                self.config.training.num_input_views = self.config.training.num_views // 2
                self.config.training.num_target_views = self.config.training.num_views // 2
                num_target_views = self.config.training.num_target_views
        
        # Decide the target view indices
        if target_has_input: 
            # Randomly sample target views across all views
            index = torch.tensor([
                random.sample(range(num_views), num_target_views)
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device) # [b, num_target_views]
        else:
            assert (
                self.config.training.num_input_views + num_target_views <= self.config.training.num_views
            ), f"We have {self.config.training.num_views} views in total, but we want to select {self.config.training.num_input_views} input views and {num_target_views} target views. This is more than the total number of views we have."
            
            index = torch.tensor([
                [self.config.training.num_views - 1 - j for j in range(num_target_views)]
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device)
            index = torch.sort(index, dim=1).values # [b, num_target_views]


        for key, value in data_batch.items():
            # The following keys are the same across the whole batch, don't change based on the view
            if key in ["scene_name", "scene_latents", "norm_type"]:
                input_dict[key] = value
                target_dict[key] = value
                continue
            input_dict[key] = value[:, :self.config.training.num_input_views, ...]

            to_expand_dim = value.shape[2:] # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
            expanded_index = index.view(index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)

            # Don't have target image supervision 
            if key == "image" and not has_target_image:                
                continue
            else:
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)
        
        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)

        return input_dict, target_dict

    
    @torch.no_grad()
    def forward(self, data_batch, has_target_image=True, target_has_input=True, compute_rays=True, 
                compute_rays_camera_frame=False, pose_to_quat=False):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.
            compute_rays_camera_frame (bool): If True, compute ray_d in camera reference frame.
            pose_to_quat (bool): If True, convert camera poses to quaternions.
                
        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
                - 'ray_d_camera_frame' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'camera_pose_quats' (torch.Tensor): Shape [b, v, 4]
                - 'camera_pose_trans' (torch.Tensor): Shape [b, v, 3]
        """
        input_dict, target_dict = self.fetch_views(data_batch, has_target_image=has_target_image, target_has_input=target_has_input)

        if compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]

                ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, device=data_batch["image"].device)
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d
        
        if compute_rays_camera_frame:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]

                _, ray_d = self.compute_rays_in_camera_frame(fxfycxcy, h=image_height, w=image_width, device=data_batch["image"].device)
                dict["ray_d_camera_frame"] = ray_d
        
        if pose_to_quat:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                rotation_matrices = c2w[:, :, :3, :3]
                translation_vectors = c2w[:, :, :3, 3]
                quats = rotation_matrix_to_quaternion(rotation_matrices)
                dict["camera_pose_quats"] = quats
                dict["camera_pose_trans"] = translation_vectors

        return input_dict, target_dict

      
class ProcessDataMapAnything(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True):
        """
        Splits the input data batch into input and target sets.
        
        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'views'
                - 'index'
                - 'scene_name' (torch.Tensor)
            target_has_input (bool): If True, target includes input views.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if (self.config.training.get("dynamic_input_view_num", False) 
            and (not self.config.inference.get("if_inference", False))):
            self.config.training.num_input_views = np.random.randint(2, 5)
        

        input_dict, target_dict = {}, {}
        # index = [] save for future use if we want to select specific views

        num_target_views, num_views, bs = self.config.training.num_target_views, len(data_batch["views"]), data_batch["index"].size(0)
        assert num_target_views < num_views, f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."
        
        # Decide the target view indices
        if target_has_input: 
            # Randomly sample target views across all views
            index = torch.tensor([
                random.sample(range(num_views), num_target_views)
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["index"].device) # [b, num_target_views]
        else:
            assert (
                self.config.training.num_input_views + num_target_views <= self.config.training.num_views
            ), f"We have {self.config.training.num_views} views in total, but we want to select {self.config.training.num_input_views} input views and {num_target_views} target views. This is more than the total number of views we have."
            
            index = torch.tensor([
                [self.config.training.num_views - 1 - j for j in range(num_target_views)]
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["index"].device)
            index = torch.sort(index, dim=1).values # [b, num_target_views]


        for key, value in data_batch.items():
            if key == "scene_name":
                input_dict[key] = value
                target_dict[key] = value
                continue
            elif key == "index":            
                input_dict[key] = value[:, :self.config.training.num_input_views, ...]

                to_expand_dim = value.shape[2:] # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
                expanded_index = index.view(index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)
            elif key == "views":
                input_dict[key] = value[:self.config.training.num_input_views]
                target_dict[key] = value[self.config.training.num_input_views:]
            else:
                raise KeyError(f'Unexpected key in data_batch: {key}')
        
        # height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        # input_dict["image_h_w"] = (height, width)
        # target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)

        return input_dict, target_dict


    
    @torch.no_grad()
    def forward(self, data_batch, has_target_image=True, target_has_input=True, compute_rays=True):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.
                
        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        input_dict, target_dict = self.fetch_views(data_batch, has_target_image=has_target_image, target_has_input=target_has_input)

        return input_dict, target_dict


def pad_images_to_max_size(images, intrinsics):
    """
    images: list of tensors [n, 3, H, W] with possibly different H/W.
    intrinsics: list of tensors [n, 4] representing fx, fy, cx, cy.
    Returns: single tensor [B, n, 3, H_max, W_max]
    """
    # Find maximum dimensions
    max_height = max(img.shape[-2] for img in images)
    max_width = max(img.shape[-1] for img in images)

    # Pad images if necessary
    padded_images = []
    padded_intrinsics = []
    for img, fxfycxcy in zip(images, intrinsics):
        h_padding = max_height - img.shape[-2]
        w_padding = max_width - img.shape[-1]

        if h_padding > 0 or w_padding > 0:
            print([img.shape[-2] for img in images], [img.shape[-1] for img in images])
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            img = F.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
            
            # Shift intrinsics (cx, cy) accordingly
            fxfycxcy[2] += pad_left
            fxfycxcy[3] += pad_top
        
        padded_images.append(img)
        padded_intrinsics.append(fxfycxcy)

    images = torch.stack(padded_images, dim=0)
    intrinsics = torch.stack(padded_intrinsics, dim=0)

    return images, intrinsics


def crop_images_to_min_size(images, intrinsics):
    """
    images: list of tensors [n, 3, H, W] with possibly different H/W.
    intrinsics: list of tensors [n, 4] representing fx, fy, cx, cy.
    Returns:
        cropped_images: tensor [B, n, 3, H_min, W_min]
        adjusted_intrinsics: tensor [B, n, 4]
    """
    # Find minimum dimensions across batch
    min_height = min(img.shape[-2] for img in images)
    min_width = min(img.shape[-1] for img in images)

    cropped_images = []
    adjusted_intrinsics = []

    for img, fxfycxcy in zip(images, intrinsics):
        H, W = img.shape[-2:]
        start_y = (H - min_height) // 2
        start_x = (W - min_width) // 2

        # Crop the image (center crop)
        cropped_img = img[..., start_y:start_y + min_height, start_x:start_x + min_width]

        # Adjust intrinsics: move principal point relative to new crop
        fxfycxcy = fxfycxcy.clone()
        fxfycxcy[..., 2] -= start_x  # shift cx
        fxfycxcy[..., 3] -= start_y  # shift cy

        cropped_images.append(cropped_img)
        adjusted_intrinsics.append(fxfycxcy)

    cropped_images = torch.stack(cropped_images, dim=0)
    adjusted_intrinsics = torch.stack(adjusted_intrinsics, dim=0)

    return cropped_images, adjusted_intrinsics


def resize_images_to_target_size(images, intrinsics, target_height, target_width):
    """
    Resize all images in a batch to target_height x target_width by cropping or padding.

    Args:
        images: list of tensors [n, 3, H, W].
        intrinsics: list of tensors [n, 4] representing fx, fy, cx, cy.
        target_height: desired height (int).
        target_width: desired width (int).

    Returns:
        resized_images: tensor [B, n, 3, target_height, target_width].
        adjusted_intrinsics: tensor [B, n, 4].
    """
    resized_images = []
    adjusted_intrinsics = []

    for img, fxfycxcy in zip(images, intrinsics):
        H, W = img.shape[-2:]
        new_img = img
        new_intrinsics = fxfycxcy.clone()

        # --- CROP if larger ---
        if H > target_height or W > target_width:
            start_y = max((H - target_height) // 2, 0)
            start_x = max((W - target_width) // 2, 0)
            new_img = new_img[..., start_y:start_y + target_height, start_x:start_x + target_width]
            new_intrinsics[..., 2] -= start_x  # adjust cx
            new_intrinsics[..., 3] -= start_y  # adjust cy

        # Update sizes after cropping
        H, W = new_img.shape[-2:]

        # --- PAD if smaller ---
        pad_height = max(target_height - H, 0)
        pad_width = max(target_width - W, 0)

        if pad_height > 0 or pad_width > 0:
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            new_img = F.pad(
                new_img,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0
            )

            # adjust intrinsics for padding
            new_intrinsics[..., 2] += pad_left
            new_intrinsics[..., 3] += pad_top

        resized_images.append(new_img)
        adjusted_intrinsics.append(new_intrinsics)

    resized_images = torch.stack(resized_images, dim=0)
    adjusted_intrinsics = torch.stack(adjusted_intrinsics, dim=0)

    return resized_images, adjusted_intrinsics


def VGGT_collate(batch):
    """
    Collate function that pads images and stacks everything else.
    """
    # Images & intrinsics -> pad
    images = [b["image"] for b in batch]
    fxfycxcy = [b["fxfycxcy"] for b in batch]
    images, fxfycxcy = pad_images_to_max_size(images, fxfycxcy)

    # Everything else -> stack/list
    c2w = torch.stack([b["c2w"] for b in batch], dim=0)
    indices = torch.stack([b["index"] for b in batch], dim=0)
    scene_names = [b["scene_name"] for b in batch]
    
    if "scene_latents" not in batch[0]:
        return {
            "image": images,
            "c2w": c2w,
            "fxfycxcy": fxfycxcy,
            "index": indices,
            "scene_name": scene_names
        }

    # Add pre-computed scene latents if enabled
    scene_latents = {
        "aggregated_tokens_list": torch.stack([b["scene_latents"]["aggregated_tokens_list"] for b in batch]),
        "patch_start_idx": torch.tensor([b["scene_latents"]["patch_start_idx"] for b in batch]),
    }

    return {
        "image": images,
        "c2w": c2w,
        "fxfycxcy": fxfycxcy,
        "index": indices,
        "scene_name": scene_names,
        "scene_latents": scene_latents,
    }


def MA_collate(batch):
    """
    Collate function that pads images and stacks everything else.
    """
    # Images & intrinsics -> pad
    images = [b["image"] for b in batch]
    fxfycxcy = [b["fxfycxcy"] for b in batch]
    images, fxfycxcy = pad_images_to_max_size(images, fxfycxcy)
    
    # Everything else -> stack/list
    c2w = torch.stack([b["c2w"] for b in batch], dim=0)
    indices = torch.stack([b["index"] for b in batch], dim=0)
    scene_names = [b["scene_name"] for b in batch]
    is_metric_scale = torch.stack([b["is_metric_scale"] for b in batch], dim=0)
    norm_type = [b["norm_type"] for b in batch]
    
    collated_batch = {
        "image": images,
        "c2w": c2w,
        "fxfycxcy": fxfycxcy,
        "index": indices,
        "scene_name": scene_names,
        "is_metric_scale": is_metric_scale,
        "norm_type": norm_type,
    }
    
    # In case of dual encoder, images will already be square -> no need to pad
    if "image_LVSM" in batch[0]:
        collated_batch['image_LVSM'] = torch.stack([b["image_LVSM"] for b in batch], dim=0)
        collated_batch['fxfycxcy_LVSM'] = torch.stack([b["fxfycxcy_LVSM"] for b in batch], dim=0)
        collated_batch['c2w_LVSM'] = torch.stack([b["c2w_LVSM"] for b in batch], dim=0)
    
    return collated_batch


def PRoPE_collate(batch):
    """
    Collate function that pads images and stacks everything else.
    """
    # Images & intrinsics -> pad
    images = [b["image"] for b in batch]
    fxfycxcy = [b["fxfycxcy"] for b in batch]
    images, fxfycxcy = resize_images_to_target_size(images, fxfycxcy, target_height=294, target_width=518)
    
    # Everything else -> stack/list
    c2w = torch.stack([b["c2w"] for b in batch], dim=0)
    indices = torch.stack([b["index"] for b in batch], dim=0)
    scene_names = [b["scene_name"] for b in batch]
    is_metric_scale = torch.stack([b["is_metric_scale"] for b in batch], dim=0)
    norm_type = [b["norm_type"] for b in batch]
    
    return {
        "image": images,
        "c2w": c2w,
        "fxfycxcy": fxfycxcy,
        "index": indices,
        "scene_name": scene_names,
        "is_metric_scale": is_metric_scale,
        "norm_type": norm_type,
    }
    
    
def MapAnything_collate(batch):
    """
    Collate function for batching views with padding and ray adjustment.
    """

    num_views = len(batch[0]['views'])
    batch_size = len(batch)

    collated_views = []

    for v in range(num_views):
        imgs, norm_types, metric_scales, quats, trans = [], [], [], [], []
        rays = []
        Hs, Ws = [], []

        for sample in batch:
            v_input = sample['views'][v]
            img = v_input['img']            # [3,H,W]
            ray = v_input['ray_directions_cam']  # [H,W,3]

            H, W = img.shape[-2:]
            Hs.append(H)
            Ws.append(W)

            imgs.append(img)
            rays.append(ray)
            norm_types.append(v_input['data_norm_type'])
            metric_scales.append(v_input['is_metric_scale'])
            quats.append(v_input['camera_pose_quats'])
            trans.append(v_input['camera_pose_trans'])

        max_H, max_W = max(Hs), max(Ws)

        # --- Pad images ---
        padded_imgs = []
        for img in imgs:
            _, H, W = img.shape
            pad_left = (max_W - W) // 2
            pad_right = max_W - W - pad_left
            pad_top = (max_H - H) // 2
            pad_bottom = max_H - H - pad_top
            padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))
            padded_imgs.append(padded_img)
        imgs = torch.stack(padded_imgs, dim=0)  # [B,3,max_H,max_W]

        # --- Pad rays (shift principal point) ---
        padded_rays = []
        for ray, H, W in zip(rays, Hs, Ws):
            pad_left = (max_W - W) // 2
            pad_right = max_W - W - pad_left
            pad_top = (max_H - H) // 2
            pad_bottom = max_H - H - pad_top

            padded_ray = torch.zeros((max_H, max_W, 3), dtype=ray.dtype, device=ray.device)
            padded_ray[pad_top:pad_top+H, pad_left:pad_left+W, :] = ray
            padded_rays.append(padded_ray)
        ray_directions = torch.stack(padded_rays, dim=0)  # [B,max_H,max_W,3]

        # --- Stack metadata ---
        metric_scales = torch.stack(metric_scales, dim=0)  # [B]
        quats = torch.stack(quats, dim=0)  # [B,4]
        trans = torch.stack(trans, dim=0)  # [B,3]

        collated_views.append({
            'img': imgs,
            'ray_directions_cam': ray_directions,
            'data_norm_type': norm_types,  # list of strings
            'is_metric_scale': metric_scales,
            'camera_pose_quats': quats,
            'camera_pose_trans': trans,
        })
    
    # Everything else -> stack/list
    indices = torch.stack([b["index"] for b in batch], dim=0)
    scene_names = [b["scene_name"] for b in batch]
    
    return {
        "views": collated_views,
        "index": indices,
        "scene_name": scene_names,
    }


def denormalize(image, norm_type="dinov2"):
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
    mean = img_norm.mean.view(1, 1, 3, 1, 1).to(image.device)
    std = img_norm.std.view(1, 1, 3, 1, 1).to(image.device)
    
    image = (image * std) + mean
    
    return image


# Fixed resolution mappings with precomputed aspect ratios as keys
RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),  # 1:1
        1.321: (518, 392),  # 4:3
        1.542: (518, 336),  # 3:2
        1.762: (518, 294),  # 16:9
        2.056: (518, 252),  # 2:1
        3.083: (518, 168),  # 3.2:1
        0.757: (392, 518),  # 3:4
        0.649: (336, 518),  # 2:3
        0.567: (294, 518),  # 9:16
        0.486: (252, 518),  # 1:2
    },
    512: {
        1.000: (512, 512),  # 1:1
        1.333: (512, 384),  # 4:3
        1.524: (512, 336),  # 3:2
        1.778: (512, 288),  # 16:9
        2.000: (512, 256),  # 2:1
        3.200: (512, 160),  # 3.2:1
        0.750: (384, 512),  # 3:4
        0.656: (336, 512),  # 2:3
        0.562: (288, 512),  # 9:16
        0.500: (256, 512),  # 1:2
    },
}

# Precomputed sorted aspect ratio keys for efficient lookup
ASPECT_RATIO_KEYS = {
    518: sorted(RESOLUTION_MAPPINGS[518].keys()),
    512: sorted(RESOLUTION_MAPPINGS[512].keys()),
}


def find_closest_aspect_ratio(aspect_ratio, resolution_set):
    """
    Find the closest aspect ratio from the resolution mappings using efficient key lookup.

    Args:
        aspect_ratio (float): Target aspect ratio
        resolution_set (int): Resolution set to use (518 or 512)

    Returns:
        tuple: (target_width, target_height) from the resolution mapping
    """
    aspect_keys = ASPECT_RATIO_KEYS[resolution_set]

    # Find the closest aspect ratio key using binary search approach
    closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

    return RESOLUTION_MAPPINGS[resolution_set][closest_key]