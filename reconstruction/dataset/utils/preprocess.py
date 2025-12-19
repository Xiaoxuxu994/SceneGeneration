import torch
import numpy as np
from torchvision import transforms as TF
from PIL import Image
from einops import rearrange

from dataset.utils.camera_reader import CustomCamera
from utils import DEVICE_NAME
import cv2


def closed_form_inverse_se3(se3: torch.Tensor, R: torch.Tensor=None, T: torch.Tensor=None, device="cuda"):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """

    # Validate shapes
    if se3.shape[-2:] != (4, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    R_transposed = R.transpose(-2, -1)
    top_right = -R_transposed @ T
    
    inverted_matrix = torch.stack([torch.eye(4) for _ in range(len(se3))], dim=0).to(device=device)
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def resize_image(img, new_width, new_height, mode="crop", interp_mode="bicubic"):
    # Resize with new dimensions (width, height)
    target_size = 518
    img = torch.nn.functional.interpolate(img[None], size=(new_height, new_width), mode=interp_mode)[0]

    # Center crop height if it's larger than 518 (only in crop mode)
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]
    
    # For pad mode, pad to make a square of target_size x target_size
    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]
        
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            
            # Pad with white (value=1.0)
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
    return img


def _read_exr(
    fname: str , fmt: str="torch"
) -> np.ndarray | torch.Tensor | Image.Image:
    """
    Reads an EXR image file using OpenCV.

    Args:
        fname (str or Path): The filename of the EXR image to read.
        fmt (str): The format of the output data. Can be one of:
            - "torch": Returns a PyTorch tensor.
            - "np": Returns a NumPy array.
            - "PIL": Returns a PIL Image object.
            Defaults to "torch".

    Returns:
        The EXR image data in the specified output format.

    Raises:
        NotImplementedError: If the specified output format is not supported.
        ValueError: If data shape is not supported, e.g. multi-channel PIL float images.

    Notes:
        The EXR image is read in its original format, without any conversion or rescaling.
    """
    data = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(f"Failed to read EXR file: {fname}")
    if fmt == "torch":
        # Convert to PyTorch tensor with float32 dtype
        data = torch.from_numpy(data).float()
    elif fmt == "np":
        # Convert to NumPy array with float32 dtype
        data = np.array(data, dtype=np.float32)
    elif fmt == "PIL":
        if data.ndim != 2:
            raise ValueError("PIL does not support multi-channel EXR images")

        # Convert to PIL Image object
        data = Image.fromarray(data)
    else:
        raise NotImplementedError(f"fmt not supported: {fmt}")
    return data


def depth_file_loader(depth_file_name: str):
    """Given the name of a depth file, return the depth map as a numpy array.

    Args:
        depth_file_name (str): Name of the depth file.
    
    Returns:
        A numpy array of shape (1, H, W) denoting the depth map.
    """
    if depth_file_name.endswith(".npy"):
        return np.load(depth_file_name, dtype=np.float32)
    elif depth_file_name.endswith(".exr"):
        return _read_exr(depth_file_name, fmt="np")
    else:
        raise NotImplementedError("Depth file type {} is not implemented.".format(depth_file_name.split('.')[-1]))


def load_and_preprocess_images(image_path_list, depth_path_list=None, mask_path_list=None, mode="crop", device="cpu"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        return torch.tensor([])
        # raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    depths = []
    masks  = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    def get_new_shape(width, height, mode, target_size):
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14
        return new_width, new_height

    # First process all images and collect their shapes
    for idx, image_path in enumerate(image_path_list):

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
        new_width, new_height = get_new_shape(width, height, mode, target_size)

        # Resize with new dimensions (width, height)
        img = to_tensor(img).to(device=device)  # Convert to tensor (0, 1)
        img = resize_image(img, new_width, new_height, mode=mode)
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        
        if depth_path_list is not None:
            depth_file = depth_path_list[idx]
            if depth_file.endswith(".npy"):
                depth = np.load(depth_file)
                depth = torch.from_numpy(depth).to(device=device)
            elif depth_file.endswith(".exr"):
                depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                depth = np.array(depth) # h, w
                depth = torch.from_numpy(depth).to(device=device)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            depth[torch.isinf(depth)] = 1000
            depth = resize_image(depth, new_width, new_height, mode=mode)
            depths.append(depth)
        
        if mask_path_list is not None:
            mask_file_outlier, mask_file_sky = mask_path_list[idx]
            mask_outlier = torch.from_numpy(np.array(cv2.imread(mask_file_outlier))[..., :1]).to(device=device).permute(2, 0, 1) # 1 h w
            mask_sky = torch.from_numpy(np.array(cv2.imread(mask_file_sky))[..., :1]).to(device=device).permute(2, 0, 1) # 1 h w
            mask = (mask_outlier == 0) & (mask_sky == 0)
            mask = resize_image(mask.to(torch.float32), new_width, new_height, mode=mode, interp_mode="nearest")
            masks.append(mask > 1e-4)

    images = torch.stack(images)  # concatenate images
    if depth_path_list is not None:
        depths = torch.stack(depths)
    if mask_path_list is not None:
        masks = torch.stack(masks)

    return images, depths if depth_path_list is not None else None, masks if mask_path_list is not None else None, (height, width)


def load_and_preprocess_images_from_meta(frame_metas: list[dict], mode="crop", device="cpu"):
    # Check for empty list
    if len(frame_metas) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    depths = []
    masks = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for idx, frame_meta in enumerate(frame_metas):

        # Open image
        img = Image.open(frame_meta["image"])

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        # img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img).to(device=device)  # Convert to tensor (0, 1)
        img = resize_image(img, new_width, new_height, mode=mode)
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        
        depth = depth_file_loader(frame_meta["mvsa_depth"])
        depth = torch.from_numpy(depth).to(device=device).unsqueeze(0)
        depth = resize_image(depth, new_width, new_height, mode=mode)
        depths.append(depth)
        
        mask = depth_file_loader(frame_meta["mvsa_depth_conf"])
        mask = torch.from_numpy(mask).to(device=device).unsqueeze(0)
        mask = resize_image(mask, new_width, new_height, mode=mode)
        mask = torch.all(torch.cat([(mask > 0.25).unsqueeze(-1), (depth > 1e-4).unsqueeze(-1)], dim=-1), dim=-1)
        masks.append(mask)

    images = torch.stack(images)  # concatenate images
    depths = torch.stack(depths)
    masks = torch.stack(masks)

    return images, depths, masks, (height, width)


def compute_rays(c2w: torch.Tensor, fxfycxcy: torch.Tensor, h=None, w=None, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pixel-wise rays from given camera informations. \n
    Reference to LVSM (https://github.com/Haian-Jin/LVSM/blob/main/utils/data_utils.py#L38)
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w].
        ray_d (torch.tensor): [b, v, 3, h, w].
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


def plucker_map_from_vector(ray_o, ray_d, method="defaul_plucker"):
    '''
    Compute plucker map from pixel-wise rays. \n
    Reference to LVSM (https://github.com/Haian-Jin/LVSM/blob/main/model/LVSM_scene_encoder_decoder.py#L189).
    Args:
        images: [b, v, c, h, w]
        ray_o: [b, v, 3, h, w]
        ray_d: [b, v, 3, h, w]
        method: Method for creating pose conditioning
    Returns:
        plucker_map: [b, v, 6, h, w]
    '''
    if method == "custom_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
        
    elif method == "aug_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
        
    else:  # default_plucker
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d], dim=2)
    
    return pose_cond


def plucker_map_from_camera(camera_list: list[CustomCamera], device="cuda"):
    """Compute plucker ray map for each camera.

    Args:
        camera_list (CustomCamera): Input camera list. 
    
    Returns:
        plucker_map (torch.Tensor): Plucker map of input cameras, tensor of shape (V, 6, H, W).
    """
    extrinsics = []
    intrinsics = []
    for camera in camera_list:
        extrinsics.append(camera.w2c_trans)
        intrinsics.append(torch.tensor([camera.intrinsic[0, 0] * camera.w, camera.intrinsic[1, 1] * camera.h, 
                                        camera.intrinsic[0, 2] * camera.w, camera.intrinsic[1, 2] * camera.h]))
    extrinsics = torch.stack(extrinsics).to(device=device)
    intrinsics = torch.stack(intrinsics).to(device=device)
    extrinsics = closed_form_inverse_se3(extrinsics, device=device)
    extrinsics = extrinsics.reshape(1, len(camera_list), 4, 4)
    intrinsics = intrinsics.reshape(1, len(camera_list), 4)
    
    ray_o, ray_d = compute_rays(extrinsics, intrinsics, device=device)
    
    plucker_map = plucker_map_from_vector(ray_o, ray_d)
    
    return plucker_map.squeeze()