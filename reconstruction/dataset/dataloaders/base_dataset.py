import os
from glob import glob
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

import torch
from torch.utils.data import Dataset

import dataset.utils.io as io
import dataset.utils.common as cm
from dataset.utils.preprocess import preprocess_images_for_encode
from utils.common import calc_extr_inv
from utils.log_utils import depth2pointmap


MAX_SAMPLE_RETRY = 5
VAL_RATIO = 0.02
IMAGE_MAX_SIZE = 518


class Base_Dataset(Dataset):
    def __init__(self, 
                 data_root, 
                 image_format=".jpg", 
                 depth_lowerbound=1e-3, 
                 depth_upperbound=65536, 
                 sample_min=2, 
                 sample_max=24, 
                 train_proc=0.0, 
                 is_train=None, 
                 batch_size=1, 
                 **kwargs, 
                 ):
        """_summary_
        Args:
            train_proc (float): Sample_view dynamicly increases in a linear manner along training process, and train_proc is an indicator in [0, 1] for this. 
        """
        super().__init__()
        assert sample_min <= sample_max, "Illegal sample_min: {} vs. sample_max: {}".format(sample_min, sample_max)
        self.data_root = data_root
        self.image_format = image_format
        self.depth_lowerbound = depth_lowerbound
        self.depth_upperbound = depth_upperbound
        self.depth_is_z = True
        self.train_proc = train_proc
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.sample_num = sample_min + int(np.round((sample_max - sample_min) * train_proc))
        self.is_train = is_train
        self.batch_size = batch_size # Ensure each batch is sampled from the same dataset to keep image shape consistency.
    
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self):
        raise NotImplementedError()

    def view_sampler(self, clip_view_num):
        """
        View sampler function. 
        Currently we regard all data set pre-determined clips, and randomly sample a view number within [self.sample_min, self.sample_max].
        """
        sample_num = min(clip_view_num, self.sample_num)
        view_ids = np.random.choice(np.arange(0, clip_view_num), sample_num, replace=False)
        return view_ids
    
    def prepare_data_for_dino_encode(self, data, max_size=518):
        data["input_images"], data["input_intrs"], cbox = preprocess_images_for_encode(
            data["input_images"], intrs=data["input_intrs"], interp_method="bilinear", max_size=max_size)
        data["input_depths"], _, _ = preprocess_images_for_encode(data["input_depths"], cbox=cbox, max_size=max_size)
        if data["input_depth_masks"] is not None:
            data["input_depth_masks"], _, _ = preprocess_images_for_encode(data["input_depth_masks"], cbox=cbox, max_size=max_size)
        if data.get("input_tracks0", None) is not None:
            ## Crop processed tracking information for realistic dataset. 
            ## For synthetic dataset, we resize images first and then calculate matching. 
            out_H, out_W = data["input_images"].shape[-2:]
            offset = torch.tensor([cbox[2], cbox[0]], dtype=data["input_tracks0"].dtype, device=data["input_tracks0"].device)[None, None] # 1 1 2
            data["input_tracks0"] = data["input_tracks0"] - offset
            data["input_tracks1"] = data["input_tracks1"] - offset
            visib_update_mask = ((data["input_tracks0"][..., 0] >= -0.5) & 
                                 (data["input_tracks0"][..., 0] < out_W - 0.5) & 
                                 (data["input_tracks0"][..., 1] >= -0.5) & 
                                 (data["input_tracks0"][..., 1] < out_H - 0.5) &
                                 (data["input_tracks1"][..., 0] >= -0.5) & 
                                 (data["input_tracks1"][..., 0] < out_W - 0.5) & 
                                 (data["input_tracks1"][..., 1] >= -0.5) & 
                                 (data["input_tracks1"][..., 1] < out_H - 0.5))
            data["input_visibs0"] = data["input_visibs0"] & visib_update_mask
            data["input_visibs1"] = data["input_visibs1"] & visib_update_mask
        
        return data
    
    def is_visib_valid(self, visib: torch.Tensor, view_num: int, min_thres=0.00625):
        """Validate visiblity of a data as if it composes a minimal_spanning_tree with its maximum covisibility above min_thres.
        Args:
            visib (torch.Tensor): Tensor of shape (view_num(view_num-1)/2, ), with each element denoting the ratio of matched pixels over overall resolution on the view.
            view_num (int): Number of views corresponds to visib. 
            min_thres (float): The minimal covisibility allowed in the spanning tree. 
        """
        g_mat = np.zeros((view_num, view_num), dtype=np.int32)
        for i in range(view_num):
            for j in range(i + 1, view_num):
                g_mat[i, j] = visib[cm.get_tracking_idx_by_pair(i, j, total=view_num)] > min_thres
        g_mat = csr_array(g_mat)
        mst = minimum_spanning_tree(g_mat)
        mst_g_mat = mst.toarray()
        nonzero_count = len(mst_g_mat.flatten().nonzero()[0])
        return nonzero_count == view_num - 1

    def update_train_proc(self, train_proc):
        self.train_proc = min(max(train_proc, 0.0), 1.0)
        self.sample_num = self.sample_min + int(np.round((self.sample_max - self.sample_min) * train_proc))


class Realistic_Dataset(Base_Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_list = self.get_sample_list()
    
    def get_sample_list(self):
        sample_list = sorted(glob(os.path.join(self.data_root, "*")))
        filtered_sample_list = []
        for sample_folder in sample_list:
            if (os.path.exists(os.path.join(sample_folder, "rgb")) and 
                os.path.exists(os.path.join(sample_folder, "moge_depth")) and 
                os.path.exists(os.path.join(sample_folder, "moge_depth_mask")) and 
                os.path.exists(os.path.join(sample_folder, "cotracker3"))):
                filtered_sample_list.append(sample_folder)
        valid_sample_num = int(len(filtered_sample_list) * VAL_RATIO)
        if self.is_train is not None and valid_sample_num > 0:
            filtered_sample_list = filtered_sample_list[:-valid_sample_num] if self.is_train else filtered_sample_list[-valid_sample_num:]
        return filtered_sample_list
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, _):
        batch = []
        for _ in range(self.batch_size):
            sample_idx = np.random.randint(low=0, high=len(self))
            batch.append(self._getitem_fn(sample_idx))
        batch = cm.customized_collate_fn(batch)
        return batch

    def _getitem_fn(self, sample_idx):
        # try:
        ## List files
        sample_folder = self.sample_list[sample_idx]
        image_list = sorted(glob(os.path.join(sample_folder, "rgb", "*{}".format(self.image_format))))
        depth_list = sorted(glob(os.path.join(sample_folder, "moge_depth", "*.exr")))
        depth_mask_list = sorted(glob(os.path.join(sample_folder, "moge_depth_mask", "*.png")))
        tracking_file = sorted(glob(os.path.join(sample_folder, "cotracker3", "*.npz")))[0]
        camera_file = os.path.join(sample_folder, sample_folder.split('/')[-1] + ".npy")
        
        ## Sample views
        clip_view_num = len(image_list)
        assert clip_view_num == len(depth_list) and clip_view_num == len(depth_mask_list), \
            "Image list length: {}, depth list length: {}, depth mask list length: {}".format(clip_view_num, len(depth_list), len(depth_mask_list))
        
        ## Drop last to adapt to batch_size that greater than 1.
        if clip_view_num < self.sample_num:
            return None
        
        extrs_raw, intrs_raw = io.read_realistic_camera_npy(camera_file)
        tracking_info_raw = io.read_cotracker_info_as_tensor(tracking_file)
        
        ## Filter out scenes with numerical illegal cameras.
        if extrs_raw.abs().max() > 1e3:
            return None
        
        ## Try at most MAX_SAMPLE_RETRY times to get a legal sample.
        current_try = 0
        while current_try < MAX_SAMPLE_RETRY:
            current_try += 1
            samp_view_ids = self.view_sampler(clip_view_num)
            # print("[ DEBUG ] view ids: {}".format(samp_view_ids))
            
            ## Load files
            images = torch.stack([io.read_image_as_tensor(image_list[idx]) for idx in samp_view_ids])
            depths = torch.stack([io.read_depth_as_tensor(depth_list[idx]) for idx in samp_view_ids])
            depth_masks = torch.stack([io.read_depth_mask_as_tensor(depth_mask_list[idx]) for idx in samp_view_ids])
            extrs = torch.stack([extrs_raw[idx] for idx in samp_view_ids], dim=0)
            intrs = torch.stack([intrs_raw[idx] for idx in samp_view_ids], dim=0)
            tracking_info = dict(track0=[], track1=[], visib0=[], visib1=[])
            for i in range(len(samp_view_ids)):
                for j in range(i + 1, len(samp_view_ids)):
                    tracking_info_idx = cm.get_tracking_idx_by_pair(samp_view_ids[i], samp_view_ids[j], total=clip_view_num)
                    if samp_view_ids[i] < samp_view_ids[j]:
                        tracking_info["track0"].append(tracking_info_raw["track0"][tracking_info_idx])
                        tracking_info["track1"].append(tracking_info_raw["track1"][tracking_info_idx])
                        tracking_info["visib0"].append(tracking_info_raw["visib0"][tracking_info_idx])
                        tracking_info["visib1"].append(tracking_info_raw["visib1"][tracking_info_idx])
                    else:
                        tracking_info["track0"].append(tracking_info_raw["track1"][tracking_info_idx])
                        tracking_info["track1"].append(tracking_info_raw["track0"][tracking_info_idx])
                        tracking_info["visib0"].append(tracking_info_raw["visib1"][tracking_info_idx])
                        tracking_info["visib1"].append(tracking_info_raw["visib0"][tracking_info_idx])
            for k in tracking_info.keys():
                tracking_info[k] = torch.stack(tracking_info[k])
            
            ## Return data as dictionary
            data = dict(
                input_images=images, # v 3 h w
                input_depths=depths, # v 1 h w
                input_depth_masks=depth_masks, # v 1 h w
                input_tracks0=tracking_info["track0"], # v(v-1)/2 k 2
                input_tracks1=tracking_info["track1"], # v(v-1)/2 k 2
                input_visibs0=tracking_info["visib0"], # v(v-1)/2 k
                input_visibs1=tracking_info["visib1"], # v(v-1)/2 k
                input_extrs=extrs, # v 4 4
                input_intrs=intrs, # v 3 3
                input_points=None, 
            )
            data = self.prepare_data_for_dino_encode(data, max_size=IMAGE_MAX_SIZE)
            
            if not self.is_visib_valid(data["input_visibs0"].sum(dim=-1) / (data["input_images"].shape[-2] * data["input_images"].shape[-1]), 
                                       view_num=len(samp_view_ids), min_thres=0.0001):
                # print("[ DEBUG - WARNING ] Retry triggered.")
                continue
                        
            return data
        
        print("[ WARNING ] Max retries exceed!")
        return None
        
        # except:
        #     return None


class Synthetic_Dataset(Base_Dataset):
    def __init__(self, clip_length: int, samp_interval: int, match_eps=0.01, **kwargs):
        super().__init__(**kwargs)
        self.clip_length = clip_length
        self.samp_interval = samp_interval
        self.match_eps = match_eps
        self.sample_list = self.get_sample_list()
        """
        Define each sample as, 
        dict(
            input_images(image file list), 
            input_depths(depth file list), 
            input_depth_masks(depth mask file list), 
            input_extrs(array of shape n,4,4), 
            input_intrs(array of shape n,3,3)
        )
        And self.sample_list is a list of samples. 
        """
    
    def create_sample_list(self):
        raise NotImplementedError()
    
    def get_sample_list(self):
        sample_list_file = os.path.join(self.data_root, "reliev3r_sample_list_c{}_s{}.npz".format(self.clip_length, self.samp_interval))
        if os.path.exists(sample_list_file):
            print("[ INFO ] Loading preprocessed sample list for {} from '{}'".format(self.__class__, sample_list_file))
            sample_list = np.load(sample_list_file, allow_pickle=True)['data']
        else:
            print("[ INFO ] Creating sample list for {}".format(self.__class__))
            sample_list = self.create_sample_list()
            np.savez_compressed(sample_list_file, data=sample_list)
        valid_sample_num = int(len(sample_list) * VAL_RATIO)
        if self.is_train is not None and valid_sample_num > 0:
            sample_list = sample_list[:-valid_sample_num] if self.is_train else sample_list[-valid_sample_num:]
        return sample_list

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, _):
        batch = []
        for _ in range(self.batch_size):
            sample_idx = np.random.randint(low=0, high=len(self))
            batch.append(self._getitem_fn(sample_idx))
        batch = cm.customized_collate_fn(batch)
        return batch
    
    def _getitem_fn(self, sample_idx):
        sample_meta = self.sample_list[sample_idx]
        image_list = sample_meta["input_images"]
        depth_list = sample_meta["input_depths"]
        depth_mask_list = sample_meta["input_depth_masks"]
        extr_list = torch.from_numpy(sample_meta["input_extrs"]).to(dtype=torch.float32)
        intr_list = torch.from_numpy(sample_meta["input_intrs"]).to(dtype=torch.float32)
        V = len(image_list)
        
        ## Drop last to adapt to batch_size that greater than 1.
        if V < self.sample_num:
            return None
        
        ## Try at most MAX_SAMPLE_RETRY times to get a legal sample.
        current_try = 0
        while current_try < MAX_SAMPLE_RETRY:
            current_try += 1
            
            samp_view_ids = self.view_sampler(clip_view_num=V)
            # print("[ DEBUG ] view ids: {}".format(samp_view_ids))
            samp_view_ids_tensor = torch.tensor(samp_view_ids, dtype=torch.int64)
            extrs = extr_list[samp_view_ids_tensor]
            intrs = intr_list[samp_view_ids_tensor]
            
            images = torch.stack([io.read_image_as_tensor(image_list[idx]) for idx in samp_view_ids])
            depths = torch.stack([io.read_depth_as_tensor(depth_list[idx]) for idx in samp_view_ids])
            if not self.depth_is_z:
                ## Depth is ray length, transform into z depth
                depths = cm.rdepth_to_zdepth(depths, intrs)
            depth_masks = torch.stack([io.read_depth_mask_as_tensor(depth_mask_list[idx]) for idx in samp_view_ids]) \
                if depth_mask_list is not None else None
            if hasattr(self, "depth_scale"):
                depths = depths * self.depth_scale
            if depth_masks is None:
                depth_masks = ~depths.isinf()
            else:
                depth_masks = depth_masks & (~depths.isinf())
            depth_masks = depth_masks & (depths < self.depth_upperbound) & (depths > self.depth_lowerbound)
            depths[~depth_masks] = depths[depth_masks].max() * 1.05
            depth_masks = depth_masks.to(depths.dtype)
            
            ## Resize and crop images to fit for dino encoder.
            data = dict(
                input_images=images, # v 3 h w
                input_depths=depths, # v 1 h w
                input_depth_masks=depth_masks, # v 1 h w
                input_extrs=extrs, # v 4 4
                input_intrs=intrs, # v 3 3
            )
            data = self.prepare_data_for_dino_encode(data, max_size=IMAGE_MAX_SIZE)
            depths, intrs = data["input_depths"], data["input_intrs"]
            
            ## Dataset extrinsics are defined as world-to-camera. Transform to camera-to-world when passed.
            tracks0, tracks1, visibs = cm.exhaustive_geo2match(
                depths, calc_extr_inv(extrs), intrs, eps=self.match_eps, samples_to_keep=10000)
            
            ## Validate if the image group is connected by checking covisibilitu.
            if not self.is_visib_valid(visibs.sum(dim=-1) / (images.shape[-2] * images.shape[-1]), 
                                       view_num=len(samp_view_ids), min_thres=0.00625):
                # print("[ DEBUG - WARNING ] Retry triggered.")
                continue
            
            data.update(dict(input_tracks0=tracks0, # v(v-1)/2 k 2
                             input_tracks1=tracks1, # v(v-1)/2 k 2
                             input_visibs0=visibs, # v(v-1)/2 k
                             input_visibs1=visibs, # v(v-1)/2 k
                             input_points=depth2pointmap(depths, extrs, intrs, device="cpu", is_intr_normalized=True), # v h w 3
            ))
            return data
        
        print("[ WARNING ] Max retries exceed!")
        return None