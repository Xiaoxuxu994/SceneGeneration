import os
import cv2
import h5py
import numpy as np
import torch
from justpfm import justpfm


def read_image_as_tensor(file_name: str, device="cpu", dtype=torch.float32):
    """
    Read RGB image, transform it into shape of (C, H, W), regularize value into [0, 1].
    """
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        image = np.array(image) / 255
        image = torch.from_numpy(image).permute(2, 0, 1)
    elif file_name.endswith(".hdf5"):
        with h5py.File(file_name, "r") as fp:
            image = fp['dataset'][()]
        image = torch.from_numpy(image).permute(2, 0, 1) / 255
    else:
        raise NotImplementedError("Not implemented image format '{}'".format(file_name.split('.')[-1]))
    assert image.ndim == 3 and image.shape[0] == 3, "Illegal image shape {}".format(image.shape)
    return image.to(device=device, dtype=dtype)


def read_depth_as_tensor(file_name: str, device="cpu", dtype=torch.float32, regard_inf_as=1e9):
    """
    Read depth maps, transform it into shape of (1, H, W).
    """
    if file_name.endswith(".exr"):
        depth = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        depth = torch.from_numpy(depth).unsqueeze(0)
    elif file_name.endswith(".npy"):
        depth = np.load(file_name)
        depth = torch.from_numpy(depth).squeeze().unsqueeze(0)
    elif file_name.endswith(".hdf5"):
        with h5py.File(file_name, "r") as fp:
            depth = fp['dataset'][()]
        depth = torch.from_numpy(depth).unsqueeze(0)
    elif file_name.endswith(".png"):
        depth = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        depth = torch.from_numpy(depth).unsqueeze(0) * 1.0
    elif file_name.endswith(".pfm"):
        depth = justpfm.read_pfm(file_name=file_name).copy()
        depth = torch.from_numpy(depth).squeeze().unsqueeze(0)
    else:
        raise NotImplementedError("Not implemented depth format '{}'".format(file_name.split('.')[-1]))
    assert depth.ndim == 3 and depth.shape[0] == 1, "Illegal depth shape {}".format(depth.shape)
    depth[depth.isinf()] = regard_inf_as
    return depth.to(device=device, dtype=dtype)


def read_depth_mask_as_tensor(file_name: str, device="cpu", dtype=torch.float32):
    """
    Read valid mask of depth map, transform it into shape of (1, H, W).
    """
    if file_name.endswith(".png"):
        mask = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        mask = np.array(mask) / 255
        mask = torch.from_numpy(mask).unsqueeze(0)
    else:
        raise NotImplementedError("Not implemented depth mask format '{}'".format(file_name.split('.')[-1]))
    assert mask.ndim == 3 and mask.shape[0] == 1, "Illegal depth shape {}".format(mask.shape)
    return mask.to(device=device, dtype=dtype)


def read_cotracker_info_as_tensor(file_name: str, device="cpu", dtype=torch.float32, dtype_visib=torch.bool):
    """
    Read preprocessed cotracker output. 
    """
    if file_name.endswith(".npz"):
        matching_infos = np.load(file_name, allow_pickle=True)["data"]
        track0 = torch.from_numpy(np.concat([matching_info["tracks"][0] for matching_info in matching_infos], axis=0)) # n k 2
        track1 = torch.from_numpy(np.concat([matching_info["tracks"][1] for matching_info in matching_infos], axis=0)) # n k 2
        visib0 = torch.from_numpy(np.concat([matching_info["visibility"][0] for matching_info in matching_infos], axis=0)) # n k
        visib1 = torch.from_numpy(np.concat([matching_info["visibility"][1] for matching_info in matching_infos], axis=0)) # n k
    else:
        raise NotImplementedError("Not implemented cotracker file format '{}'".format(file_name.split('.')[-1]))
    assert track0.ndim == 3 and visib0.ndim == 2 and track0.shape[:2] == visib0.shape
    return dict(
        track0=track0.to(device=device, dtype=dtype), 
        track1=track1.to(device=device, dtype=dtype), 
        visib0=visib0.to(device=device, dtype=dtype_visib),
        visib1=visib1.to(device=device, dtype=dtype_visib), 
    )


def read_realistic_camera_npy(file_name, device="cpu", dtype=torch.float32):
    """
    Read preprocessed packed camera parameters.
    """
    if file_name.endswith(".npy"):
        cameras = np.load(file_name, allow_pickle=True).item()
        extrs = cameras["extr_mat"] # v 4 4
        intrs = cameras["intr_mat"] # v 3 3
    else:
        raise NotImplementedError("Not implemented camera file format '{}'".format(file_name.split('.')[-1]))
    assert extrs.ndim == 3 and extrs.shape[-2] == 4 and extrs.shape[-1] == 4
    assert intrs.ndim == 3 and intrs.shape[-2] == 3 and intrs.shape[-1] == 3
    return torch.from_numpy(extrs).to(device=device, dtype=dtype), \
           torch.from_numpy(intrs).to(device=device, dtype=dtype)
