import os
from glob import glob
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import dataset.utils.io as io
import dataset.utils.common as cm
from dataset.utils.preprocess import preprocess_images_for_encode, crop_images_and_intrs, crop_tracks_and_visibs
from utils.common import calc_extr_inv, normalize_trans, depth2pointmap, globalpoint2local, trans_geo_to_first_frame_and_norm
from utils.loss_utils import samp_points_by_pixel

MAX_SAMPLE_RETRY = 10
VAL_RATIO = 0.02
VIEW_OVERLAP_THRE_REALISTIC_DATA = 0.30
VIEW_OVERLAP_THRE_SYNTHETIC_DATA = 0.20


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
                 depth_is_z=True, 
                 pred_z_depth=True, 
                 crop_lowerbound=0.0, 
                 crop_upperbound=0.2,
                 img_max_size=518,
                 patch_size=14,
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
        self.depth_is_z = depth_is_z
        self.pred_z_depth = pred_z_depth
        self.train_proc = train_proc
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.sample_num = sample_min + int(np.round((sample_max - sample_min) * train_proc))
        self.is_train = is_train
        self.batch_size = batch_size # Ensure each batch is sampled from the same dataset to keep image shape consistency.
        self.crop_lowerbound = crop_lowerbound
        self.crop_upperbound = crop_upperbound
        self.img_max_size = img_max_size
        self.img_max_sizes = [518, 504, 490, 476, 462, 448, 434, 420]
        self.patch_size = patch_size
    
    def __len__(self):
        raise NotImplementedError()
    
    def view_sampler(self, clip_view_num):
        """
        View sampler function. 
        Currently we regard all data set pre-determined clips, and randomly sample a view number within [self.sample_min, self.sample_max].
        """
        sample_num = min(clip_view_num, self.sample_num)
        view_ids = np.random.choice(np.arange(0, clip_view_num), sample_num, replace=False)
        return view_ids
    
    def view_sampler_iterative(self, visibs: torch.Tensor, clip_view_num: int, thre: float):
        """
        Sample views of a sample from a clip in an iterative approach.
        Given a sampled view set S from a clip C, each time add a view x in C-S, which has a minimal maximum covisibility with views in S that \
            above a given threshold. In this way, the given threshold represents sparcity of the sampled views.

        Args:
            visibs (torch.Tensor): Covisibility of pair-wise tracking pixels. Tensor of shape (V(V-1)/2, gs*gs)
            clip_viwe_num (int): Number of views in the clip.
        """
        visib_mat = cm.deflat_visibility_mat(visibs, clip_view_num) # v v, upper-triangle matrix.
        visib_mat[visib_mat < thre] = 0
        
        view_ids = []
        sampled_mask = torch.zeros(clip_view_num, dtype=torch.bool)
        # first_view_id = legal_roots[np.random.randint(low=0, high=len(legal_roots))]
        first_view_id = np.random.randint(low=0, high=clip_view_num)
        view_ids.append(first_view_id)
        sampled_mask[first_view_id] = True
        sampled_view_cnt = 1
        while sampled_view_cnt < self.sample_num:
            _visib_mat = visib_mat.clone()
            _visib_mat[sampled_mask] = 0
            _visib_mat[:, ~sampled_mask] = 0
            _visib_vec = _visib_mat.max(dim=-1).values # v
            _visib_vec[_visib_vec == 0] = 2
            new_view_id = _visib_vec.min(dim=0).indices
            if _visib_vec[new_view_id] == 2:
                return None
            else:
                view_ids.append(new_view_id.item())
                sampled_mask[new_view_id] = True
                sampled_view_cnt += 1
        return np.array(view_ids)
    
    def prepare_data_for_dino_encode(self, data):
        max_image_size = self.img_max_sizes[np.random.randint(0, 8)]
        data["input_images"], data["input_intrs"], cbox, scale_factor = preprocess_images_for_encode(
            data["input_images"], intrs=data["input_intrs"], interp_method="bilinear", max_size=max_image_size, patch_size=self.patch_size)
        data["input_depths"], _, _, _ = preprocess_images_for_encode(data["input_depths"], cbox=cbox, max_size=max_image_size, patch_size=self.patch_size)
        if data["input_depth_masks"] is not None:
            data["input_depth_masks"], _, _, _ = preprocess_images_for_encode(data["input_depth_masks"], cbox=cbox, max_size=max_image_size, patch_size=self.patch_size)
        if data.get("input_tracks0", None) is not None:
            ## Crop processed tracking information for realistic dataset. 
            ## For synthetic dataset, we resize images first and then calculate matching. 
            out_H, out_W = data["input_images"].shape[-2:]
            data["input_tracks0"] = data["input_tracks0"] * scale_factor
            data["input_tracks1"] = data["input_tracks1"] * scale_factor
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
    
    def aug_by_crop(self, data, crop_ratio=None):
        if crop_ratio is None:
            return data
        images = data.get("input_images", None)
        intrs  = data.get("input_intrs", None)
        assert images is not None
        raw_shape = (images.shape[-2], images.shape[-1])
        images, intrs, cropped_shape = crop_images_and_intrs(images, intrs=intrs, scale=crop_ratio)
        data["input_images"], data["input_intrs"] = images, intrs
        if data.get("input_depths", None) is not None:
            data["input_depths"], _, _ = crop_images_and_intrs(data["input_depths"], out_shape=cropped_shape)
        if data.get("input_depth_masks", None) is not None:
            data["input_depth_masks"], _, _ = crop_images_and_intrs(data["input_depth_masks"], out_shape=cropped_shape)
        if data.get("input_tracks0", None) is not None:
            data["input_tracks0"], data["input_visibs0"] = \
                crop_tracks_and_visibs(data["input_tracks0"], data["input_visibs0"], raw_shape, cropped_shape)
            data["input_tracks1"], data["input_visibs1"] = \
                crop_tracks_and_visibs(data["input_tracks1"], data["input_visibs1"], raw_shape, cropped_shape)
        return data
    
    def __getitem__(self, _):
        batch = []
        max_retries = 2
        crop_ratio = (np.random.rand() * (self.crop_upperbound - self.crop_lowerbound) + self.crop_lowerbound, 
                      np.random.rand() * (self.crop_upperbound - self.crop_lowerbound) + self.crop_lowerbound) # Augmentation by cropping.
        tries = 0
        while tries < max_retries:
            sample_idx = np.random.randint(low=0, high=len(self))
            batch = self._getitem_fn(sample_idx, crop_ratio=crop_ratio, batch_size=self.batch_size)
            if batch is not None:
                break
            tries += 1
        try:
            """
            try can fail when the batch is illegal, specifically:
            1. 'tries' >= 'max_retris' but 'data' found None.
            2. Inconsistent image resolution within the batch.
            """
            batch = cm.customized_collate_fn(batch)
            return batch
        except:
            return None
        
    def is_visib_valid(self, visib: torch.Tensor, view_num: int, min_thres=6.25e-3):
        """Validate visiblity of a data as if it composes a minimal_spanning_tree with its maximum covisibility above min_thres.
        Args:
            visib (torch.Tensor): Tensor of shape (view_num(view_num-1)/2, ), with each element denoting the ratio of matched pixels over overall resolution on the view.
            view_num (int): Number of views corresponds to visib. 
            min_thres (float): The minimal covisibility allowed in the spanning tree. 
        """
        g_mat = np.zeros((view_num, view_num), dtype=np.float32)
        for i in range(view_num):
            for j in range(i + 1, view_num):
                pair_idx = cm.get_tracking_idx_by_pair(i, j, total=view_num)
                if visib[pair_idx] > min_thres:
                    g_mat[i, j] = -visib[pair_idx]
        g_mat = csr_array(g_mat)
        mst = minimum_spanning_tree(g_mat)
        mst_g_mat = mst.toarray()
        nonzero_count = len(mst_g_mat.flatten().nonzero()[0])
        if nonzero_count == view_num - 1:
            match_view_i, match_view_j = [], []
            # for nonzero_idx in mst_g_mat.flatten().nonzero()[0]:
            #     match_view_i.append(nonzero_idx // view_num)
            #     match_view_j.append(nonzero_idx % view_num)
            for i in range(view_num):
                for j in range(i + 1, view_num):
                    match_view_i.append(i)
                    match_view_j.append(j)
            match_view_i = np.array(match_view_i)
            match_view_j = np.array(match_view_j)
            return match_view_i, match_view_j
        return None

    def update_train_proc(self, train_proc):
        self.train_proc = min(max(train_proc, 0.0), 1.0)
        self.sample_num = self.sample_min + int(np.round((self.sample_max - self.sample_min) * self.train_proc))


class Synthetic_Dataset(Base_Dataset):
    def __init__(self, clip_length: int, samp_interval: int, match_eps=0.01, overlap_thre_synth=0.15, **kwargs):
        super().__init__(**kwargs)
        self.clip_length = clip_length
        self.samp_interval = samp_interval
        self.match_eps = match_eps
        self.overlap_thre = overlap_thre_synth
        self.sample_list = self.get_sample_list()
        print("[ INFO ] {} samples found for {} as {}.".format(
            len(self), self.__class__.__name__,"training set" if self.is_train else "validation set"))
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
        if self.__class__.__name__ == "ASE_Dataset":
            sample_list_file = sample_list_file + ".update.npz"
        if os.path.exists(sample_list_file):
            print("[ INFO ] Loading preprocessed sample list for {} from '{}'".format(self.__class__.__name__, sample_list_file))
            sample_list = np.load(sample_list_file, allow_pickle=True)
            if 'data' in sample_list.files:
                sample_list = sample_list['data']
            else:
                self.data_list = sample_list
                sample_list = sample_list.files
        else:
            print("[ INFO ] Creating sample list for {}".format(self.__class__.__name__))
            sample_list = self.create_sample_list()
            np.savez_compressed(sample_list_file, data=sample_list)
        valid_sample_num = max(int(len(sample_list) * VAL_RATIO), 1)
        if self.is_train is not None and valid_sample_num > 0:
            sample_list = sample_list[:-valid_sample_num] if self.is_train else sample_list[-valid_sample_num:]
        return sample_list

    def __len__(self):
        return len(self.sample_list) #// 10 * 2
    
    def fetch_meta(self, sample_idx):
        if hasattr(self, "data_list"):
            return self.data_list[self.sample_list[sample_idx]].item()
        else:
            return self.sample_list[sample_idx]
    
    def filter_top_depth(self, data, p=0.04):
        depths = data["input_depths"] * data["input_depth_masks"]
        V, _, H, W = depths.shape
        depths = depths.reshape(V, -1)
        thres = torch.topk(depths, k=int(H * W * p * data["input_depth_masks"].mean()) + 1, dim=-1).values.min(dim=-1).values # V
        valid_mask = (data["input_depths"] < thres[..., None, None, None]).to(depths.dtype) # V 1 H W
        data["input_depth_masks"] = data["input_depth_masks"] * valid_mask
        return data
    
    def _getitem_fn(self, sample_idx, crop_ratio=None, batch_size=1):
        try:
            timer = cm.MyTimer()
            timer.record_start("0_Total")
            sample_meta = self.fetch_meta(sample_idx)
            image_list = sample_meta["input_images"]
            depth_list = sample_meta["input_depths"]
            depth_mask_list = sample_meta["input_depth_masks"]
            extr_list = torch.from_numpy(sample_meta["input_extrs"]).to(dtype=torch.float32).clone()
            intr_list = torch.from_numpy(sample_meta["input_intrs"]).to(dtype=torch.float32).clone()
            V = len(image_list)
            
            ## Drop last to adapt to batch_size that greater than 1.
            if V < self.sample_num:
                print("[ WARNING ] Error datas from insufficient view num from {}".format(self.__class__.__name__))
                return None
            
            timer.record_start("1_IO")
            target_size = (480, 640) if self.__class__.__name__ == "ScanNetv2_Dataset" else None
            all_images = torch.stack([io.read_image_as_tensor(f, target_size) for f in image_list])
            all_depths = torch.stack([io.read_depth_as_tensor(f, regard_inf_as=65536) for f in depth_list])
            all_depth_masks = torch.stack([io.read_depth_mask_as_tensor(f) for f in depth_mask_list]) \
                if depth_mask_list is not None else None
            timer.record_end("1_IO")
            
            timer.record_start("2_Preprocess")
            if not self.depth_is_z:
                ## Depth is ray length, transform into z depth
                all_depths = cm.rdepth_to_zdepth(all_depths, intr_list)
            if hasattr(self, "depth_scale"):
                all_depths = all_depths * self.depth_scale
            if all_depth_masks is None:
                all_depth_masks = ~all_depths.isinf()
            else:
                all_depth_masks = all_depth_masks & (~all_depths.isinf())
            all_depth_masks = all_depth_masks & (all_depths < self.depth_upperbound) & (all_depths > self.depth_lowerbound)
            all_depths[~all_depth_masks] = all_depths[all_depth_masks].max() * 1.05
            all_depth_masks = all_depth_masks.to(all_depths.dtype)
            all_data = dict(
                input_images=all_images, # v 3 h w
                input_depths=all_depths, # v 1 h w
                input_depth_masks=all_depth_masks, # v 1 h w
                input_extrs=extr_list, # v 4 4
                input_intrs=intr_list, # v 3 3
                dataset_name=self.__class__.__name__, 
            )
            if crop_ratio is not None:
                all_data = self.aug_by_crop(all_data, crop_ratio=crop_ratio)
            all_data = self.prepare_data_for_dino_encode(all_data)
            all_data = self.filter_top_depth(all_data, p=0.004)
            all_images = all_data["input_images"]
            all_depths = all_data["input_depths"]
            all_depth_masks = all_data["input_depth_masks"]
            extr_list  = all_data["input_extrs"]
            intr_list  = all_data["input_intrs"]
            timer.record_end("2_Preprocess")
            
            ## Try at most MAX_SAMPLE_RETRY times to get a legal sample.
            current_try = 0
            batch_data = []
            timer.record_start("4_exhaustive_matching")
            grid_size = 100
            _, _, visibs_all = cm.exhaustive_geo2match(all_depths, calc_extr_inv(extr_list), intr_list, eps=self.match_eps, grid_size=grid_size)
            visibs_all = visibs_all.to(dtype=torch.float32).mean(dim=-1) # v(v-1)/2
            timer.record_end("4_exhaustive_matching")
            while current_try < MAX_SAMPLE_RETRY and len(batch_data) < batch_size:
                current_try += 1
                
                # samp_view_ids = self.view_sampler(clip_view_num=V)
                samp_view_ids = self.view_sampler_iterative(visibs_all, clip_view_num=V, thre=self.overlap_thre)
                if samp_view_ids is None:
                    continue
                
                samp_view_ids_tensor = torch.tensor(samp_view_ids, dtype=torch.int64)
                images = all_images[samp_view_ids_tensor]
                depths = all_depths[samp_view_ids_tensor]
                depth_masks = all_depth_masks[samp_view_ids_tensor]
                extrs = extr_list[samp_view_ids_tensor]
                intrs = intr_list[samp_view_ids_tensor]
                
                ## Resize and crop images to fit for dino encoder.
                data = dict(
                    input_images=images, # v 3 h w
                    input_depths=depths, # v 1 h w
                    input_depth_masks=depth_masks, # v 1 h w
                    input_extrs=extrs, # v 4 4
                    input_intrs=intrs, # v 3 3
                    dataset_name=self.__class__.__name__, 
                )
                
                timer.record_start("3_MatchCalc")
                ## Dataset extrinsics are defined as world-to-camera. Transform to camera-to-world when passed.
                tracks0, tracks1, visibs = cm.exhaustive_geo2match(
                    depths, calc_extr_inv(extrs), intrs, eps=self.match_eps, samples_to_keep=10000, grid_size=grid_size)
                ## Validate if the image group is connected by checking covisibilitu.
                matched_pairs = self.is_visib_valid(visibs.sum(dim=-1) / (grid_size * grid_size), 
                                                    view_num=len(samp_view_ids), min_thres=0.0)
                timer.record_end("3_MatchCalc")
                assert matched_pairs is not None, "This should not happen."
                if matched_pairs is None:
                    continue
                matched_view_i, matched_view_j = matched_pairs
                data.update(dict(input_tracks_view_id0=torch.from_numpy(matched_view_i), # v-1
                                 input_tracks_view_id1=torch.from_numpy(matched_view_j), # v-1
                                 input_tracks0=torch.stack([tracks0[cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                                            for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k 2
                                 input_tracks1=torch.stack([tracks1[cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                                            for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k 2
                                 input_visibs0=torch.stack([visibs[cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                                            for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k
                                 input_visibs1=torch.stack([visibs[cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                                            for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k
                                 input_points=depth2pointmap(depths, extrs, intrs, device="cpu", is_intr_normalized=True), # v h w 3
                                 ))
                if not self.pred_z_depth:
                    data.update(dict(input_depths=cm.zdepth_to_rdepth(depths, intrs)))
                
                ## Normalize geometrics.
                # if sample_idx <= len(self) // 2:
                data["input_points"], data["input_extrs"], data["input_depths"] = \
                    trans_geo_to_first_frame_and_norm(data["input_points"], data["input_extrs"], data["input_depths"])
                data["input_points_local"] = globalpoint2local(data["input_points"], data["input_extrs"])
                # else:
                #     data["input_points"] = None
                #     data["input_points_local"] = None
                
                batch_data.append(data)
            
            if len(batch_data) == batch_size:
                timer.record_end("0_Total")
                timer.report()
                return batch_data
            # print("[ WARNING ] Max retries exceed!")
            # print("[ WARNING ] Error data from max retries from {}".format(self.__class__.__name__))
            return None

        except:
            return None


class Realistic_Dataset(Base_Dataset):
    def __init__(self, overlap_thre_real=0.25, **kwargs):
        super().__init__(**kwargs)
        self.clip_length = 32
        self.sample_list = self.get_sample_list()
        self.overlap_thre = overlap_thre_real
        print("[ INFO ] {} samples found for {} as {}.".format(
            len(self), self.__class__.__name__,"training set" if self.is_train else "validation set"))
    
    def get_sample_list(self):
        # sample_list_file = os.path.join(self.data_root, "reliev3r_sample_list.npz")
        sample_list_file = os.path.join(self.data_root, "reliev3r_sample_list_c{}.npz".format(self.clip_length))
        if os.path.exists(sample_list_file):
            print("[ INFO ] Loading preprocessed sample list for {} from '{}'".format(self.__class__.__name__, sample_list_file))
            filtered_sample_list = np.load(sample_list_file, allow_pickle=True)["data"]
        else:
            print("[ INFO ] Creating sample list for {}".format(self.__class__.__name__))
            sample_list = sorted(glob(os.path.join(self.data_root, "sub_*", "*")))
            filtered_sample_list = []
            for sample_folder in tqdm(sample_list):
                if not io.run_check(sample_folder, 32):
                    continue
                if (os.path.exists(os.path.join(sample_folder, "rgb")) and 
                    os.path.exists(os.path.join(sample_folder, "moge_depth")) and 
                    os.path.exists(os.path.join(sample_folder, "moge_depth_mask")) and 
                    os.path.exists(os.path.join(sample_folder, "romav2"))):
                    filtered_sample_list.append(sample_folder)
            np.savez_compressed(sample_list_file, data=filtered_sample_list)
            
        valid_sample_num = int(len(filtered_sample_list) * VAL_RATIO)
        if self.is_train is not None and valid_sample_num > 0:
            filtered_sample_list = filtered_sample_list[:-valid_sample_num] if self.is_train else filtered_sample_list[-valid_sample_num:]
        return filtered_sample_list
    
    def update_track_visib_by_depth_mask(self, track: torch.Tensor, visib: torch.Tensor, depth_mask: torch.Tensor):
        """Filter out illegal tracking points from visib by identifying the illegal values in depth mask.

        Args:
            track (torch.Tensor): Tensor of shape (k, 2).
            visib (torch.Tensor): Tensor of shape (k).
            depth_mask (torch.Tensor): Tensor of shape (1, H, W).
        """
        valid_track = samp_points_by_pixel(depth_mask[None], track[None], mode="nearest", padding_mode="zeros")[0, :, 0] # k
        valid_track = valid_track > 0.5
        valid_track = valid_track & visib
        return valid_track
    
    def __len__(self):
        return len(self.sample_list)

    def _getitem_fn(self, sample_idx, crop_ratio=None, batch_size=1):
        try:
            timer = cm.MyTimer()
            timer.record_start("0_Total")
            timer.record_start("1_IO")
            
            ## List files
            sample_folder = self.sample_list[sample_idx]
            image_list = sorted(glob(os.path.join(sample_folder, "rgb", "*{}".format(self.image_format))))
            depth_list = sorted(glob(os.path.join(sample_folder, "moge_depth", "*.exr")))
            depth_mask_list = sorted(glob(os.path.join(sample_folder, "moge_depth_mask", "*.png")))
            # tracking_file = sorted(glob(os.path.join(sample_folder, "cotracker3", "*.npz")))
            tracking_file = sorted(glob(os.path.join(sample_folder, "romav2", "*.npz")))
            if len(tracking_file) == 0:
                return None
            tracking_file = tracking_file[0]
            if self.__class__.__name__ == "CO3Dv2_Dataset":
                camera_file_name = sample_folder.split('/')[-1] + ".npy"
                camera_file = os.path.join("/efs-mi-east4-3/z00818731/co3dv2-cams", camera_file_name)
            else:
                camera_file = os.path.join(sample_folder, sample_folder.split('/')[-1] + ".npy")
            if not os.path.exists(camera_file):
                return None
            
            ## Sample views
            clip_view_num = len(image_list)
            if clip_view_num != len(depth_list) or clip_view_num != len(depth_mask_list):
                return None
            
            ## Drop last to adapt to batch_size that greater than 1.
            if clip_view_num < self.sample_num:
                return None
            
            invert_extr = (self.__class__.__name__ == "ScanNet_Dataset" or self.__class__.__name__ == "ScanNetpp_Dataset")
            # extrs_raw, intrs_raw = io.read_realistic_camera_npy(camera_file, inv_extr=invert_extr)
            extrs_raw, intrs_raw = io.read_realistic_camera_npy(camera_file)
            if invert_extr:
                extrs_raw = calc_extr_inv(extrs_raw)
            tracking_info_raw = io.read_cotracker_info_as_tensor(tracking_file)
            if self.__class__.__name__ == "WildRGBD_Dataset":
                intrs_raw[..., 0, 2] = 0.5
                intrs_raw[..., 1, 2] = 0.5
                intrs_raw[..., 2, 0] = 0.0
                intrs_raw[..., 2, 1] = 0.0
            
            ## Filter out scenes with numerical illegal cameras.
            try:
                if extrs_raw.abs().max() > 1e3:
                    return None
            except:
                return None
            if len(extrs_raw) != clip_view_num:
                return None
            
            all_images = torch.stack([io.read_image_as_tensor(f) for f in image_list])
            all_depths = torch.stack([io.read_depth_as_tensor(f, regard_inf_as=1e3) for f in depth_list])
            all_depth_masks = torch.stack([io.read_depth_mask_as_tensor(f) for f in depth_mask_list]) \
                if depth_mask_list is not None else None
            
            timer.record_end("1_IO")
            timer.record_start("2_Preprocess")
            
            all_data = dict(
                input_images=all_images, # v 3 h w
                input_depths=all_depths, # v 1 h w
                input_depth_masks=all_depth_masks, # v 1 h w
                input_tracks0=tracking_info_raw["track0"], # v(v-1)/2 k 2
                input_tracks1=tracking_info_raw["track1"], # v(v-1)/2 k 2
                input_visibs0=tracking_info_raw["visib0"], # v(v-1)/2 k
                input_visibs1=tracking_info_raw["visib1"], # v(v-1)/2 k
                input_extrs=extrs_raw, # v 4 4
                input_intrs=intrs_raw, # v 3 3
                input_points=None, 
                dataset_name=self.__class__.__name__, 
            )
            if crop_ratio is not None:
                all_data = self.aug_by_crop(all_data, crop_ratio=crop_ratio)
            all_data = self.prepare_data_for_dino_encode(all_data)
            all_images = all_data["input_images"]
            all_depths = all_data["input_depths"]
            all_depth_masks = all_data["input_depth_masks"]
            tracking_info_raw["track0"] = all_data["input_tracks0"]
            tracking_info_raw["track1"] = all_data["input_tracks1"]
            tracking_info_raw["visib0"] = all_data["input_visibs0"]
            tracking_info_raw["visib1"] = all_data["input_visibs1"]
            intrs_raw = all_data["input_intrs"]
            
            timer.record_end("2_Preprocess")
            
            ## Try at most MAX_SAMPLE_RETRY times to get a legal sample.
            current_try = 0
            batch_data = []
            visibs_all = torch.logical_and(tracking_info_raw["visib0"], 
                                           tracking_info_raw["visib1"]).to(dtype=torch.float32).mean(dim=-1) # v(v-1)/2
            while current_try < MAX_SAMPLE_RETRY and len(batch_data) < batch_size:
                current_try += 1
                # samp_view_ids = self.view_sampler(clip_view_num)
                samp_view_ids = self.view_sampler_iterative(visibs_all, clip_view_num, self.overlap_thre)
                if samp_view_ids is None:
                    continue
                
                ## Load files
                samp_view_ids_tensor = torch.tensor(samp_view_ids, dtype=torch.int64)
                images = all_images[samp_view_ids_tensor]
                depths = all_depths[samp_view_ids_tensor]
                depth_masks = all_depth_masks[samp_view_ids_tensor]
                extrs = extrs_raw[samp_view_ids_tensor]
                intrs = intrs_raw[samp_view_ids_tensor]
                tracking_info = dict(track0=[], track1=[], visib0=[], visib1=[])
                for i in range(len(samp_view_ids)):
                    for j in range(i + 1, len(samp_view_ids)):
                        tracking_info_idx = cm.get_tracking_idx_by_pair(samp_view_ids[i], samp_view_ids[j], total=clip_view_num)
                        track0 = tracking_info_raw["track0"][tracking_info_idx]
                        track1 = tracking_info_raw["track1"][tracking_info_idx]
                        visib0 = tracking_info_raw["visib0"][tracking_info_idx]
                        visib1 = tracking_info_raw["visib1"][tracking_info_idx]
                        visib0 = self.update_track_visib_by_depth_mask(track0, visib0, depth_masks[i])
                        visib1 = self.update_track_visib_by_depth_mask(track1, visib1, depth_masks[j])
                        visib = visib0 & visib1
                        if samp_view_ids[i] < samp_view_ids[j]:
                            tracking_info["track0"].append(track0)
                            tracking_info["track1"].append(track1)
                            tracking_info["visib0"].append(visib)
                            tracking_info["visib1"].append(visib)
                        else:
                            tracking_info["track0"].append(track1)
                            tracking_info["track1"].append(track0)
                            tracking_info["visib0"].append(visib)
                            tracking_info["visib1"].append(visib)
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
                    input_points_local=None, 
                    dataset_name=self.__class__.__name__, 
                )
                
                timer.record_start("3_MatchCalc")
                matched_pairs = self.is_visib_valid(data["input_visibs0"].sum(dim=-1) / 625, 
                                                    view_num=len(samp_view_ids), min_thres=0.0)
                timer.record_end("3_MatchCalc")
                if matched_pairs is None:
                    continue
                matched_view_i, matched_view_j = matched_pairs
                data.update(dict(input_tracks_view_id0=torch.from_numpy(matched_view_i), # v-1
                                 input_tracks_view_id1=torch.from_numpy(matched_view_j), # v-1
                                 input_tracks0=torch.stack([data["input_tracks0"][cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                               for i, j in zip(matched_view_i, matched_view_j)], dim=0), # v-1 k 2
                                 input_tracks1=torch.stack([data["input_tracks1"][cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                               for i, j in zip(matched_view_i, matched_view_j)], dim=0), # v-1 k 2
                                 input_visibs0=torch.stack([data["input_visibs0"][cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                               for i, j in zip(matched_view_i, matched_view_j)], dim=0), # v-1 k
                                 input_visibs1=torch.stack([data["input_visibs1"][cm.get_tracking_idx_by_pair(i, j, len(samp_view_ids))] \
                                               for i, j in zip(matched_view_i, matched_view_j)], dim=0), # v-1 k
                                 ))
                if not self.pred_z_depth:
                    data.update(dict(input_depths=cm.zdepth_to_rdepth(data["input_depths"], data["input_intrs"])))
                data["input_extrs"] = normalize_trans(data["input_extrs"])
                batch_data.append(data)
            
            if len(batch_data) == batch_size:
                timer.record_end("0_Total")
                timer.report()
                return batch_data
            return None
        
        except:
            return None

