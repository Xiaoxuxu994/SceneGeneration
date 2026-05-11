import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from einops import rearrange
from utils.platform import DEVICE_NAME

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

class CameraHead(nn.Module):
    def __init__(self, dim=512, pred_fov=False, input_register_token_num=None):
        super().__init__()
        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim) if input_register_token_num is None \
                else nn.Linear(output_dim*(input_register_token_num+1),output_dim*(input_register_token_num+1)),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim) if input_register_token_num is None \
                else nn.Linear(output_dim*(input_register_token_num+1),output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)
        self.zero_4x4 = torch.zeros((4, 4), device=DEVICE_NAME)
        self.pred_fov = pred_fov
        self.input_register_token_num = input_register_token_num
        if self.pred_fov:
            self.fc_fov = nn.Linear(output_dim, 2)

    def forward(self, feat, patch_h, patch_w, v=None, register_tokens=None):
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())              ##########
        feat = feat.view(feat.size(0), -1)
        
        if register_tokens is not None and self.input_register_token_num is not None:
            register_tokens = register_tokens.reshape(register_tokens.size(0), -1)
            feat = torch.cat([feat, register_tokens], dim=-1)

        feat = self.more_mlps(feat)  # [B, D_]
        with torch.amp.autocast(device_type=DEVICE_NAME, enabled=False):
            out_t = self.fc_t(feat.float())  # [B,3]
            out_r = self.fc_rot(feat.float())  # [B,9]
            pose = self.convert_pose_to_4x4(BN, out_r, out_t, feat.device, v=v)
            if self.pred_fov:
                out_fov = torch.sigmoid(self.fc_fov(feat.float())) * torch.pi # [B, 2]
                return pose, out_fov
            return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device, v=None):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.stack([self.zero_4x4 for _ in range(B)])
        if v is not None:
            pose  = rearrange(pose, "(b v) n m -> b v n m", v=v)          # b v 4 4
            out_r = rearrange(out_r, "(b v) n m -> b v n m", v=v)         # b v 3 3
            out_t = rearrange(out_t, "(b v) n -> b v n", v=v)             # b v 3
        pose[..., :3, :3] = out_r
        pose[..., :3, 3]  = out_t
        pose[..., 3, 3]   = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        def det_3x3(mat):
            return (mat[..., 0, 0] * mat[..., 1, 1] * mat[..., 2, 2] + 
                    mat[..., 0, 1] * mat[..., 1, 2] * mat[..., 2, 0] + 
                    mat[..., 0, 2] * mat[..., 1, 0] * mat[..., 2, 1] - 
                    mat[..., 0, 2] * mat[..., 1, 1] * mat[..., 2, 0] - 
                    mat[..., 0, 0] * mat[..., 1, 2] * mat[..., 2, 1] - 
                    mat[..., 0, 1] * mat[..., 1, 0] * mat[..., 2, 2])
        # det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        det = det_3x3(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r