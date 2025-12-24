import numpy as np
import torch

file = np.load('/efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole/supervise_view_fix32_viewInfo_normV14/ff17657f71_viewInfo.npz')
scale =  torch.tensor(file['scale']).float()
T_norm = torch.tensor(file['T_norm']).float()
ori_w2c = torch.tensor(file['ori_extrinsics_norm128_v14_fix32']).float()
normed_w2c = torch.tensor(file ['extrinsics_Tnorm_norm128_v14_fix32']).float()
K = torch.tensor(file['normed_intrinsics_norm128_v14_fix32']).float()
H, W = torch.tensor(file['ori_image']).shape[-2:]
K[:,0,:] *= W
K[:,1,:] *= H

def transform_w2c_trans_then_scale(w2c, s, T):
    # w2c: [N,4,4]
    # s: scalar
    # T: [3] or [N,3], translation applied BEFORE scaling (X' = s*(X + T))
    single = False
    if w2c.ndim == 2:
        w2c = w2c.unsqueeze(0); single = True
    N = w2c.shape[0]
    device = w2c.device

    if T.ndim == 1:
        T = T.unsqueeze(0).expand(N, -1)

    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]

    t_new = s * t - s * torch.matmul(R, T.unsqueeze(-1)).squeeze(-1)
    # == s * (t - R @ T)
    w2c_new = w2c.clone()
    w2c_new[:, :3, :3] = R
    w2c_new[:, :3, 3] = t_new
    return w2c_new[0] if single else w2c_new

def update_w2c_after_mesh_transform(w2c, center, scale):
    """
    w2c: [N, 4, 4] world-to-camera matrices
    center: [3] numpy or tensor, the mesh bounding box center
    scale: float, mesh extent / 2
    """

    device = w2c.device
    center = torch.tensor(center, dtype=torch.float32, device=device)

    R = w2c[:, :3, :3]                         # [N,3,3]
    t = w2c[:, :3, 3]                          # [N,3]

    # 1. compute camera center C = -R^T t
    C = -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)     # [N,3]

    # 2. apply world transform: C' = (C - center) / scale
    C_new = (C - center) / scale

    # 3. new t' = -R * C'
    t_new = -(R @ C_new.unsqueeze(-1)).squeeze(-1)

    # assemble new w2c
    w2c_new = w2c.clone()
    w2c_new[:, :3, :3] = R
    w2c_new[:, :3, 3] = t_new

    return w2c_new

s = 1/scale
T = T_norm[:3,3]
new_w2c = transform_w2c_trans_then_scale(ori_w2c, s, T)
center = -T * scale 
new_w2c_trans = update_w2c_after_mesh_transform(ori_w2c, center, scale)

idx = 0
print(new_w2c[idx])
print('-------')
print(new_w2c_trans[idx])
print('-------')
print(normed_w2c[idx])

print('-'*60)


import torch

def apply_mesh_transform(points, center, scale):
    """
    points: [M,3]
    transform: X' = (X - center) / scale
    """
    return (points - center) / scale


def project(K, Xc):
    """Xc: [M,3], return [M,2]"""
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    u = K[0,0] * x + K[0,2]
    v = K[1,1] * y + K[1,2]
    return torch.stack([u, v], dim=-1)


def verify_camera_transform(w2c_old, w2c_new, K, pts, center, scale):
    """
    w2c_old / w2c_new: [4,4]
    pts: [M,3]
    """
    R_old, t_old = w2c_old[:3,:3], w2c_old[:3,3]
    R_new, t_new = w2c_new[:3,:3], w2c_new[:3,3]

    # 1. project original points
    Xc_old = (R_old @ pts.T + t_old.unsqueeze(1)).T   # [M,3]

    # 2. transform mesh and project with new camera
    pts_new = apply_mesh_transform(pts, center, scale)
    Xc_new = (R_new @ pts_new.T + t_new.unsqueeze(1)).T

    # 3. relation check: Xc_new ≈ Xc_old / scale
    diff_world = torch.norm(Xc_new - Xc_old/scale, dim=-1)
    print("camera-space transform error mean =", diff_world.mean())

    # 4. pixel projection check
    uv_old = project(K, Xc_old)
    uv_new = project(K, Xc_new)

    diff_pix = torch.norm(uv_new - uv_old, dim=-1)
    print("pixel projection error mean =", diff_pix.mean())

    return diff_world, diff_pix

import torch

def verify_camera_transform_batch(w2c_old, w2c_new, K, pts, center, scale):
    """
    w2c_old / w2c_new: [N,4,4] world-to-camera matrices (N cameras)
    pts: [M,3] world points
    center: [3] mesh center
    scale: float
    """

    N = w2c_old.shape[0]
    M = pts.shape[0]

    device = pts.device

    R_old = w2c_old[:, :3, :3]    # [N,3,3]
    t_old = w2c_old[:, :3, 3]     # [N,3]

    R_new = w2c_new[:, :3, :3]    # [N,3,3]
    t_new = w2c_new[:, :3, 3]     # [N,3]

    # 1. original points in camera coords
    # pts: [M,3] -> [1,M,3], broadcast to [N,M,3]
    pts_b = pts.unsqueeze(0)               # [1,M,3]
    t_old_b = t_old.unsqueeze(1)           # [N,1,3]

    Xc_old = torch.matmul(pts_b, R_old.transpose(1,2)) + t_old_b   # [N,M,3]

    # 2. transform mesh points
    pts_new = (pts - center) / scale
    t_new_b = t_new.unsqueeze(1)
    Xc_new = torch.matmul(pts_new.unsqueeze(0), R_new.transpose(1,2)) + t_new_b  # [N,M,3]

    # 3. check camera-space relation
    diff_world = torch.norm(Xc_new - Xc_old / scale, dim=-1)   # [N,M]
    print("camera-space transform error mean =", diff_world.mean().item())

    # 4. project to pixels
    # K: [3,3] or [N,3,3]
    if K.ndim == 2:
        K = K.unsqueeze(0).expand(N, -1, -1)  # [N,3,3]

    def project_batch(K, Xc):
        """
        Xc: [N,M,3]
        K: [N,3,3]
        returns uv: [N,M,2]
        """
        x = Xc[...,0] / Xc[...,2]
        y = Xc[...,1] / Xc[...,2]
        u = K[:,0,0:1] * x + K[:,0,2:3]
        v = K[:,1,1:2] * y + K[:,1,2:3]
        return torch.stack([u.squeeze(-1), v.squeeze(-1)], dim=-1)  # [N,M,2]

    uv_old = project_batch(K, Xc_old)
    uv_new = project_batch(K, Xc_new)
    diff_pix = torch.norm(uv_new - uv_old, dim=-1)   # [N,M]

    print("pixel projection error mean =", diff_pix.mean().item())

    return diff_world, diff_pix


# dummy data
pts = torch.randn(1000, 3, device=ori_w2c.device) * 2 * scale
# verify_camera_transform(
#     ori_w2c, new_w2c_trans, K,
#     pts,
#     center=center,
#     scale=scale
# )
diff_world, diff_pix = verify_camera_transform_batch(
    ori_w2c, new_w2c_trans, K, pts, 
    center=center, scale=scale
) 

# EPS = 1e-6
# def check_rotation_orthonormality(w2c, atol=1e-6):
#     # w2c: [N,4,4] or [4,4]
#     if w2c.ndim == 2:
#         w2c = w2c.unsqueeze(0)
#     R = w2c[:, :3, :3]   # [N,3,3]
#     I = torch.eye(3, device=R.device).unsqueeze(0).expand_as(R)
#     should_be_I = torch.matmul(R, R.transpose(1,2))
#     err = (should_be_I - I).abs().max(dim=-1)[0].max(dim=-1)[0]  # per batch
#     det = torch.det(R)
#     return err, det  # err 标量越小越好（接近0），det 应接近 +1
# err, det = check_rotation_orthonormality(normed_w2c)
# print(err, det)
# print('-------')
# err, det = check_rotation_orthonormality(new_w2c)
# print(err, det)

# print('-'*60)

# def c2w_from_w2c(w2c):
#     # invert homogeneous 4x4
#     return torch.inverse(w2c)

# # batch version:
# def camera_centers_world_batch(w2c):
#     R = w2c[:,:3,:3]
#     t = w2c[:,:3,3].unsqueeze(-1)          # [N,3,1]
#     C = -torch.matmul(R.transpose(1,2), t).squeeze(-1)
#     return C   # [N,3]

