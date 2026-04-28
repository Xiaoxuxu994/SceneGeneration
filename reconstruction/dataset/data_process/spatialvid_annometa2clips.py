import os
import time
from glob import glob
import cv2
import numpy as np
import torch
import einops
from scipy.sparse import csr_matrix
from argparse import ArgumentParser


# VIDEO_PATH = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-260410/SpatialVID_sample/videos"
# ANNO_PATH  = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-260410/SpatialVID_sample/annotations"
# OUTPUT_PATH = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-260410/SpatialVID_sample/reliev3r_clips"
VIDEO_PATH = os.environ["SPATIALVID_VIDEO_PATH"]
ANNO_PATH = os.environ["SPATIALVID_ANNO_PATH"]
OUTPUT_PATH = os.environ["SPATIALVID_OUTPUT_PATH"]
RELIEV3R_CLIP_SIZE = 32
OUT_RESOLUTION_H = 288
OUT_RESOLUTION_W = 512


def expand(loaded_data):
    """
    Reconstruct 3D mask from sparse matrix data.
    
    Args:
        loaded_data (dict): Dictionary containing sparse matrix data with keys:
            - 'shape': Original matrix dimensions
            - 'f_{i}_data': Sparse matrix data for frame i
            - 'f_{i}_indices': Sparse matrix indices for frame i  
            - 'f_{i}_indptr': Sparse matrix index pointers for frame i
    
    Returns:
        np.ndarray: 3D array with shape (frames, height, width)
    """
    reconstructed_sparse_matrices = []
    num_frames = (len(loaded_data) - 1) // 3  # Calculate number of frames
    matrix_shape = loaded_data['shape']  # Get original matrix dimensions

    # Reconstruct sparse matrix for each frame
    for i in range(num_frames):
        data = loaded_data[f'f_{i}_data']
        indices = loaded_data[f'f_{i}_indices']
        indptr = loaded_data[f'f_{i}_indptr']
        reconstructed_matrix = csr_matrix((data, indices, indptr), shape=matrix_shape)
        reconstructed_sparse_matrices.append(reconstructed_matrix)

    # Stack all frames into a 3D array (frames, height, width)
    reconstructed_mask_3d = np.stack([m.toarray() for m in reconstructed_sparse_matrices], axis=0)
    return reconstructed_mask_3d


def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

    Args:
        quaternions: Quaternion tensor [..., 4] (order: i, j, k, r)
        eps: Small value for numerical stability

    Returns:
        Rotation matrices [..., 3, 3]
    """

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def pose_from_quaternion(pose):
    """
    Convert pose from quaternion representation to transformation matrix.

    Args:
        pose: Pose tensor [..., 7] where first 3 elements are translation (t)
              and last 4 elements are quaternion rotation (r)

    Returns:
        w2c_matrix: World-to-camera transformation matrices [..., 4, 4]
    """
    # Input is w2c, pose(n,7) or (n,v,7), output is (N,4,4) w2c matrix
    # Tensor format from https://github.com/pointrix-project/Geomotion/blob/6ab0c364f1b44ab4ea190085dbf068f62b42727c/geomotion/model/cameras.py#L6
    if type(pose) == np.ndarray:
        pose = torch.tensor(pose)
    if len(pose.shape) == 1:
        pose = pose[None]
    quat_t = pose[..., :3]  # Translation
    quat_r = pose[..., 3:]  # Quaternion rotation
    w2c_matrix = torch.zeros((*list(pose.shape)[:-1], 4, 4), device=pose.device)
    w2c_matrix[..., :3, 3] = quat_t
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)
    w2c_matrix[..., 3, 3] = 1.0
    return w2c_matrix


def intr_from_meta(meta):
    meta = torch.from_numpy(meta)
    intrs = torch.zeros(*meta.shape[:-1], 3, 3)
    intrs[..., 0, 0] = meta[..., 0]
    intrs[..., 1, 1] = meta[..., 1]
    intrs[..., 0, 2] = meta[..., 2]
    intrs[..., 1, 2] = meta[..., 3]
    intrs[..., 2, 2] = 1.0
    return intrs


def collect_scenes(video_path, anno_path):
    scene_videos = sorted(glob(os.path.join(video_path, "group_*", "*-*-*-*-*.mp4")))
    scene_names = []
    for scene_video in scene_videos:
        scene_name = "group_" + scene_video.split("group_")[-1].split('.')[0]
        if os.path.exists(os.path.join(anno_path, scene_name)):
            scene_names.append(scene_name)
    return scene_names


def main(args):
    scene_names = collect_scenes(VIDEO_PATH, ANNO_PATH)
    scene_names = scene_names[args.proc_id::args.total_proc]
    print("[ INFO ] {} scenes found for conversion.".format(len(scene_names)))
    
    total_time = 0.0
    processed_scenes = 0
    for scene_idx, scene_name in enumerate(scene_names):
        scene_video_file = os.path.join(VIDEO_PATH, scene_name) + ".mp4"
        scene_anno_folder = os.path.join(ANNO_PATH, scene_name)
        scene_extr_file = os.path.join(scene_anno_folder, "poses.npy")
        scene_intr_file = os.path.join(scene_anno_folder, "intrinsics.npy")
        scene_sub_idx = int(scene_video_file.split('/')[-2].split('_')[-1])
        flag_file = os.path.join(scene_anno_folder, "PROCESSED")
        if os.path.exists(flag_file):
            continue
        
        start_time = time.time()
        try:
            with open(os.path.join(scene_anno_folder, "indexes.txt"), "r") as fp:
                lines = fp.readlines()
                scene_frame_indices = [int(line.split(' ')[-1]) for line in lines[1:]]
        except:
            print("[ WARNING ] Missing file: {}".format(os.path.join(scene_anno_folder, "indexes.txt")))
            continue
        try:
            extrs_meta = np.load(scene_extr_file)
            intrs_meta = np.load(scene_intr_file)
            extrs = pose_from_quaternion(extrs_meta).numpy()
            intrs = intr_from_meta(intrs_meta).numpy()
        except:
            print("[ WARNING ] Missing camera files in {}".format(scene_anno_folder))
            continue

        dyn_mask_file = os.path.join(scene_anno_folder, "dyn_masks.npz")
        dyn_masks = (expand(np.load(dyn_mask_file)) > 0).astype(np.uint8) * 255
        dyn_masks = torch.from_numpy(dyn_masks[..., None, :, :])
        dyn_masks = torch.nn.functional.interpolate(dyn_masks, (OUT_RESOLUTION_H, OUT_RESOLUTION_W))
        dyn_masks = dyn_masks[..., 0, :, :].numpy()

        cap = cv2.VideoCapture(scene_video_file)
        if not cap.isOpened():
            print("[ WARNING ] Opening video {} failed.".format(scene_video_file))
            continue
        frame_idx = 0
        stored_frame_cnt = 0
        while True:
            clip_cnt = stored_frame_cnt // RELIEV3R_CLIP_SIZE
            if len(scene_frame_indices) < (clip_cnt + 1) * RELIEV3R_CLIP_SIZE:
                break
            scene_clip_output_path = os.path.join(OUTPUT_PATH, "sub_{:06d}".format(scene_sub_idx), scene_name.split('/')[-1] + "_{:05d}".format(clip_cnt))
            scene_clip_output_img_path = os.path.join(scene_clip_output_path, "rgb")
            scene_clip_output_dyn_mask_path = os.path.join(scene_clip_output_path, "dyn_mask")
            os.makedirs(scene_clip_output_img_path, exist_ok=True)
            os.makedirs(scene_clip_output_dyn_mask_path, exist_ok=True)
            
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in scene_frame_indices:
                stored_frame_cnt += 1
                frame = cv2.resize(frame, (OUT_RESOLUTION_W, OUT_RESOLUTION_H))
                cv2.imwrite(os.path.join(scene_clip_output_img_path, "frame_{:05d}.jpg".format(stored_frame_cnt)), frame)
                cv2.imwrite(os.path.join(scene_clip_output_dyn_mask_path, "frame_{:05d}.jpg".format(stored_frame_cnt)), dyn_masks[stored_frame_cnt-1][..., None])
                if stored_frame_cnt % RELIEV3R_CLIP_SIZE == 0:
                    clip_extrs = extrs[clip_cnt * RELIEV3R_CLIP_SIZE: (clip_cnt + 1) * RELIEV3R_CLIP_SIZE]
                    clip_intrs = intrs[clip_cnt * RELIEV3R_CLIP_SIZE: (clip_cnt + 1) * RELIEV3R_CLIP_SIZE]
                    clip_camera_file = os.path.join(scene_clip_output_path, os.path.basename(scene_clip_output_path) + ".npy")
                    np.save(clip_camera_file, dict(extr_mat=clip_extrs, intr_mat=clip_intrs), allow_pickle=True)
            frame_idx += 1
        cap.release()
        end_time = time.time()
        total_time += (end_time - start_time) / 3600
        processed_scenes += 1
        with open(flag_file, "w+") as fp:
            fp.write("Transfered to clips.")
        print("[ INFO ] {}/{} scenes processed, ~{:.2f} hours left.".format(
            scene_idx + 1, len(scene_names), 
            total_time / processed_scenes * (len(scene_names) - (scene_idx + 1))))
    
    print("[ INFO ] Complete.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--proc_id", type=int, default=None)
    parser.add_argument("--total_proc", type=int, default=None)
    args = parser.parse_args()
    main(args)