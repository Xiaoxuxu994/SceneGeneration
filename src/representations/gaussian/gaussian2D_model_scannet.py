import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
# import utils3d


class Gaussian2D:
    def __init__(
            self, 
            # aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        self.init_params = {
            # 'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        # self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).cuda()

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        scales = torch.clamp(scales, min=1e-4, max=0.3)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        return self._xyz # * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        # print(f'self._features_dc: {self._features_dc.shape, self._features_dc.min(), self._features_dc.max()}')
        # print(f'self._features_rest: {self._features_rest}')
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])        
        # return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc[:,:,:3].shape[1]*self._features_dc[:,:,:3].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_dc[:,:,3:].shape[1]*self._features_dc[:,:,3:].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        
        features = self._features_dc.detach().view(xyz.shape[0],-1,3)
        features_dc = features[:,0,:]
        features_rest = features[:,1:,:]

        f_dc = features_dc.contiguous().cpu().numpy() # _features_dc.shape [N, 1, 3]
        R = features_rest[:, :, 0]  # [N, 15]
        G = features_rest[:, :, 1]
        B = features_rest[:, :, 2]
        f_rest = torch.cat([R, G, B], dim=1).cpu().numpy() # _features_dc.shape [N, SH_coef-1, 3]

        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(transform, rotation)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest,  opacities, scale, rotation), axis=1) # 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        
        print("2DGS load ply")
        print(f"self.sh: {self.sh_degree}")
        
        if transform is not None:
            transform = np.array(transform)

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        xyz = np.matmul(xyz, transform)
        self._xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))        
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

        features_dc = np.zeros((xyz.shape[0], 1, 3))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            C = (self.sh_degree+1)**2 - 1
            R = features_extra[:, :C]
            G = features_extra[:, C:2*C]
            B = features_extra[:, 2*C:]
            # 拼回 [N, C, 3]
            features_extra = np.stack([R, G, B], axis=2)  # [N, C, 3]
            # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        else:
            features_extra = None
        features = features_dc if features_extra is None else np.concatenate([features_dc, features_extra], axis=1) # shape: [N, C+1, 3]
        self._features_dc = torch.tensor(features, dtype=torch.float, device=self.device).reshape(-1, (self.sh_degree+1)**2 * 3)       
            
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        self._scaling = self.inverse_scaling_activation(torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)) - self.scale_bias

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        if transform is not None:
            rots = utils3d.numpy.quaternion_to_matrix(rots)
            rots = np.matmul(transform.T, rots)
            rots = utils3d.numpy.matrix_to_quaternion(rots)
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        rots = rots - self.rots_bias[None, :]
        self._rotation = rots
        
        print(f'self._xyz: {self._xyz.shape, self._xyz.min(), self._xyz.max()}')
        print(f'self._features_dc: {self._features_dc.shape, self._features_dc.min(), self._features_dc.max()}')
        print(f'self._scaling: {self._scaling.shape, self._scaling.min(), self._scaling.max()}')
        print(f'self._rotation: {self._rotation.shape, self._rotation.min(), self._rotation.max()}')
        print(f'self._opacity: {self._opacity.shape, self._opacity.min(), self._opacity.max()}')
