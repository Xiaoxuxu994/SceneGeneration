import os
from glob import glob
from tqdm import tqdm
import h5py
import numpy as np

from dataset.dataloaders.base_dataset import Synthetic_Dataset


class HyperSim_Dataset(Synthetic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[ INFO ] {} samples found for HyperSim dataset as {}.".format(
            len(self), "training set" if self.is_train else "validation set"))
        
    def create_sample_list(self):
        ## Fetch scene list
        scene_list = sorted(glob(os.path.join(self.data_root, "ai*")))
        sample_list = list()
        for scene_id, scene in enumerate(tqdm(scene_list)):
            # if scene_id > 0:
            #     break
            ## Fetch trajectory folders by existing image folders.
            traj_list = [folder.split('_')[-3] for folder in sorted(glob(os.path.join(scene, "scene_cam_*_final_hdf5")))]
            scene_image_folders = []
            scene_depth_folders = []
            scene_extrs_files = []
            for traj_id in traj_list:
                scene_image_folder = os.path.join(scene, "scene_cam_{}_final_hdf5".format(traj_id))
                scene_depth_folder = os.path.join(scene, "scene_cam_{}_geometry_hdf5".format(traj_id))
                scene_extrs_file = os.path.join(scene, "cam_{}_extrinsics.hdf5".format(traj_id))
                if (os.path.exists(scene_image_folder) and
                    os.path.exists(scene_depth_folder) and 
                    os.path.exists(scene_extrs_file)):
                    scene_image_folders.append(scene_image_folder)
                    scene_depth_folders.append(scene_depth_folder)
                    scene_extrs_files.append(scene_extrs_file)
            
            ## Create data clips for each dataset.
            for image_folder, depth_folder, extrs_file in zip(scene_image_folders, scene_depth_folders, scene_extrs_files):
                ## Skip if mismatch found.
                if image_folder.split("_")[-3] != depth_folder.split("_")[-3] or image_folder.split("_")[-3] != extrs_file.split("_")[-2]:
                    continue
                image_list = sorted(glob(os.path.join(image_folder, "frame.*.color.hdf5")))
                depth_list = sorted(glob(os.path.join(depth_folder, "frame.*.depth_meters.hdf5")))
                if not os.path.exists(extrs_file):
                    continue
                with h5py.File(extrs_file, "r") as fp:
                    extrs = fp['dataset'][()] # n 4 3
                    extrs = np.concatenate([extrs, 
                                            np.stack([np.array([[0.0, 0.0, 0.0, 1.0]]) for _ in range(len(extrs))])], axis=-2) # n 4 4
                
                ## Regard all intrinsics as the same.
                intr = np.array([[0.8660253745, 0, 0.5], [0, 1.1547004995, 0.5], [0, 0, 1]])
                
                image_ids = [image_file.split('.')[-3] for image_file in image_list]
                
                image_clip_list = []
                depth_clip_list = []
                extr_clip_list  = []
                for image_id in image_ids[::self.samp_interval]:
                    image_file = os.path.join(image_folder, "frame.{}.color.hdf5".format(image_id))
                    depth_file = os.path.join(depth_folder, "frame.{}.depth_meters.hdf5".format(image_id))
                    if not (os.path.exists(image_file) and os.path.exists(depth_file)):
                        continue
                    image_clip_list.append(image_file)
                    depth_clip_list.append(depth_file)
                    extr_clip_list.append(extrs[int(image_id)])
                    if (len(image_clip_list) + 1) * self.samp_interval > self.clip_length:
                        sample_list.append(dict(
                            input_images=image_clip_list, 
                            input_depths=depth_clip_list, 
                            input_depth_masks=None, 
                            input_extrs=np.stack(extr_clip_list), # world to camera transform
                            input_intrs=np.stack([intr for _ in range(self.clip_length)]), 
                        ))
                        image_clip_list = []
                        depth_clip_list = []
                        extr_clip_list = []
                if len(image_clip_list) > 0:
                    sample_list.append(dict(
                        input_images=image_clip_list, 
                        input_depths=depth_clip_list, 
                        input_depth_masks=None, 
                        input_extrs=np.stack(extr_clip_list), 
                        input_intrs=np.stack([intr for _ in range(self.clip_length)]), 
                    ))
        
        return sample_list


if __name__ == '__main__':
    hypersim_root = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-251125/Reliev3R_data_sample/Hypersim_Sample"
    
    dataset = HyperSim_Dataset(sample_min=8, 
                               sample_max=8, 
                               clip_length=16, 
                               samp_interval=1, 
                               data_root=hypersim_root)
    
    import os
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from functools import partial
    from dataset.utils.common import customized_collate_fn, worker_init_fn
    from dataset.utils.vis import single_depth_intr_to_pcd, depth_maps_to_pcd, vis_batch_data
    
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=customized_collate_fn, 
                            num_workers=2, 
                            persistent_workers=True, 
                            multiprocessing_context="spawn", 
                            worker_init_fn=partial(worker_init_fn, device_id=0))
    
    for idx, data in enumerate(dataloader):
        print("[ DEBUG ] load success~")
        print("          image shape: {}, image range: {}->{}".format(
            data["input_images"].shape, data["input_images"].min().item(), data["input_images"].max().item()))
        print("          depth shape: {}, depth range: {}->{}".format(
            data["input_depths"].shape, data["input_depths"].min().item(), data["input_depths"].max().item()))
        # print("          depth mask shape: {}, depth mask range: {}->{}".format(
        #     data["input_depth_masks"].shape, data["input_depth_masks"].min().item(), data["input_depth_masks"].max().item()))
        print("          tracking shape: {}, tracking range: {}->{}".format(
            data["input_tracks1"].shape, data["input_tracks1"].min().item(), data["input_tracks1"].max().item()))
        print("          extr shape: {}".format(data["input_extrs"].shape))
        print("          intr shape: {}".format(data["input_intrs"].shape))
        print("[ DEBUG ] extrs: {}".format(data["input_extrs"]))
        print("[ DEBUG ] intrs: {}".format(data["input_intrs"]))
        B, V, K = data["input_visibs0"].shape
        print("[ DEBUG ] valid matching ratio: {}/{}".format(data["input_visibs0"].sum(), B * V * K))
        print("[ DEBUG ] valid keypoint range: {}->{}".format(
            data["input_tracks1"][data["input_visibs1"]].min(), 
            data["input_tracks1"][data["input_visibs1"]].max()
        ))
        # import time
        # time.sleep(2000)
        vis_img = vis_batch_data(data, keypoint_radius=1)
        torchvision.utils.save_image(vis_img, os.path.join(hypersim_root, "vis_samp_{:03}.jpg".format(idx)))
        
        single_depth_intr_to_pcd(os.path.join(hypersim_root, "vis_samp_{:03d}_single.ply".format(idx)), 
                                 data["input_images"][0, 0].permute(1, 2, 0), 
                                 data["input_depths"][0, 0].permute(1, 2, 0)[..., 0], 
                                 data["input_intrs"][0, 0])
        depth_maps_to_pcd(
            output_path=os.path.join(hypersim_root, "vis_samp_{:03d}_all.ply".format(idx)), 
            images=data["input_images"][0].permute(0, 2, 3, 1), 
            depths=data["input_depths"][0].permute(0, 2, 3, 1)[..., 0], 
            extrs=torch.linalg.inv(data["input_extrs"][0]), 
            intrs=data["input_intrs"][0], 
        )
    
    print("[ INFO ] Done. ")