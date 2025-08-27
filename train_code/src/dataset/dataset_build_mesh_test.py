import os
import torch
from torch.utils.data import Dataset
from src.dataset.transform import *
import open3d as o3d 

class DatasetForScanNetV6TESTSDF(Dataset):
    def __init__(self, data_root: str, data_list_name: str, **kwargs):
        """
        Args:
            data_root: Path to root directory containing .pt data
            data_list_name: File containing list of sample paths (relative or absolute)
            kwargs:

        """

        self.data_root = data_root
        with open(data_list_name, 'r') as f:
            self.samples = [os.path.join(self.data_root, p.strip()) for p in f.readlines()] ## ply
        
        self.vs_res = kwargs.get("vs_res",[128])
        aug_cfg = kwargs.get("transform",{})
        self.transforms = []
        # initialize data aug 
        if aug_cfg:
            # GridSample
            if aug_cfg.get('GridSample', {}).get('enabled', False):
                self.transforms.append(GridSample(
                    grid_size = aug_cfg['GridSample'].get('grid_size',0.05),
                    hash_type = aug_cfg['GridSample'].get('hash_type', 'fnv'),
                    mode = aug_cfg['GridSample'].get('mode','train'),
                    keys = aug_cfg['GridSample'].get('keys', ("coord",)),
                    return_grid_coord = aug_cfg['GridSample'].get('return_grid_coord', False),
                    return_min_coord = aug_cfg['GridSample'].get('return_min_coord', False),
                ))

            # UniformDownSample
            if aug_cfg.get('UniformDownSample', {}).get('enabled', False):
                self.transforms.append(UniformDownSample(
                    point_max = aug_cfg['UniformDownSample'].get('point_max',50000),
                    sample_rate = aug_cfg['UniformDownSample'].get('sample_rate', None),
                    random = aug_cfg['UniformDownSample'].get('random',True)
                ))

            # Decoderquery sample
            if aug_cfg.get('DecodeQuerySampleSDF', {}).get('enabled', False):
                self.transforms.append(DecodeQuerySampleSDF(
                    sample_rate = aug_cfg['DecodeQuerySampleSDF'].get('sample_rate',0.1),
                    max_number = aug_cfg['DecodeQuerySampleSDF'].get('max_number', 200000),
                    vs_list = aug_cfg['DecodeQuerySampleSDF'].get('vs_list',[128]),
                    near_thresh = aug_cfg['DecodeQuerySampleSDF'].get('near_thresh',0.05),
                    near_ratio = aug_cfg['DecodeQuerySampleSDF'].get('near_ratio',0.8),
                ))

            # To tensor
            if aug_cfg.get('ToTensor', {}).get('enabled', False):
                self.transforms.append(ToTensor())

            # collect_fn
            if aug_cfg.get('Collect', {}).get('enabled', False):
                self.transforms.append(Collect(
                    keys=aug_cfg['Collect'].get('keys'),
                    offset_keys_dict=aug_cfg['Collect'].get('offset_keys_dict'),
                ))

        ##  cache input query center and grid
        self.grid_index_dict = {}
        self.grid_minbound_dict = {}
        vs_list = self.vs_res if isinstance(self.vs_res, list) else [self.vs_res]
        for vs in vs_list:
            grid_idx, minbound = self._generate_dense_grid_index_and_minbound(vs)
            self.grid_index_dict[vs] = grid_idx
            self.grid_minbound_dict[vs] = minbound

    @staticmethod
    def _generate_dense_grid_index_and_minbound(resolution):
        """
        Args:
            resolution: int, number of divisions along one axis (e.g. 128)
        Returns:
            grid_idx: (N, 3), int32
            minbound_coords: (N, 3), float32
        """
        vs_size = 2.0 / resolution
        num_voxels = resolution  # extend half voxel

        origin = np.array([-1.0] * 3, dtype=np.float32)

        # 构造 index 网格
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(num_voxels),
            np.arange(num_voxels),
            np.arange(num_voxels),
            indexing='ij'
        )
        grid_idx = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.int32)

        # 计算 voxel 中心坐标
        minbound_coords = origin + (grid_idx) * vs_size

        return grid_idx, minbound_coords.astype(np.float32)

    def normalize_vertices(self,mesh,boxmin=-1,boxmax=1,boxscale=0.95):
        import open3d as o3d
        # normalize vertices
        verts = np.array(mesh.vertices)
        shifts = (verts.max(axis=0) + verts.min(axis=0)) / 2
        verts -= shifts
        scale = (boxmax - boxmin) / (verts.max(axis=0) - verts.min(axis=0)).max()
        scale *= boxscale
        verts *= scale
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        return mesh
        
    def prepare_train_data(self, input_dict):
        
        ## get query center and grid idx online
        if isinstance(self.vs_res,list):
            for vs in self.vs_res:
                input_dict[f'pc_query_grid_{vs}'] = self.grid_index_dict[vs]
                input_dict[f'pc_query_{vs}'] = self.grid_minbound_dict[vs]
        
        for transform in self.transforms:
            input_dict = transform(input_dict)
        return input_dict

    def save_vertices_as_ply_open3d(self,vertices, filepath):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector((vertices + 1) / 2)
        o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)

    def __getitem__(self, index):
        ## deal with ply
        input_mesh = o3d.io.read_triangle_mesh(self.samples[index])
        mesh_norm = self.normalize_vertices(input_mesh,boxmin=-1,boxmax=1,boxscale=0.95)

        ## fps
        coords = np.asarray(mesh_norm.vertices, dtype=np.float32)
        input_dict = {'coord':coords,
                      'scene_id':os.path.splitext(os.path.basename(self.samples[index]))[0]}
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)