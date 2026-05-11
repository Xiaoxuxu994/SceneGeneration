import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base_dataset import Synthetic_Dataset
from .tartanair_dataset import pos_quat2SE


def get_trans_device_from_camera():
    trans_device_from_camera = np.array(
        [[ 0.99606003, -0.04388682,  0.07706079, -0.0075301 ],
         [ 0.08210934,  0.78468796, -0.61442889, -0.01090855],
         [-0.03350334,  0.61833547,  0.78519983, -0.00359806],
         [ 0.        ,  0.        ,  0.        ,  1.        ]])
    rot90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    trans_device_from_camera[:3, :3] = trans_device_from_camera[:3, :3] @ rot90
    return trans_device_from_camera


class ASE_Dataset(Synthetic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_scale = 0.001
        self.depth_is_z = False
    
    def create_sample_list(self):
        scene_list = [os.path.join(self.data_root, "{}".format(i)) for i in range(100000)]
        sample_list = []
        trans_device_from_camera = get_trans_device_from_camera()
        intr = np.array([[0.42278125, 0, 0.5], [0, 0.42278125, 0.5], [0, 0, 1]])
        
        for scene in tqdm(scene_list):
            if not os.path.exists(scene):
                continue
            image_file_list = sorted(glob(os.path.join(scene, "undistort_rgb", "*.jpg")))
            depth_file_list = sorted(glob(os.path.join(scene, "undistort_depth", "*.png")))
            extr_file = os.path.join(scene, "trajectory.csv")
            extr_list = []
            try:
                extr_csv_meta = pd.read_csv(extr_file)
            except:
                continue
            extr_list = [
                pos_quat2SE([
                    extr_csv_meta["tx_world_device"][i], 
                    extr_csv_meta["ty_world_device"][i], 
                    extr_csv_meta["tz_world_device"][i], 
                    extr_csv_meta["qx_world_device"][i], 
                    extr_csv_meta["qy_world_device"][i], 
                    extr_csv_meta["qz_world_device"][i], 
                    extr_csv_meta["qw_world_device"][i], 
                ]) for i in range(len(extr_csv_meta))
            ]
            
            if not (len(image_file_list) <= len(extr_list) and 
                    len(depth_file_list) <= len(extr_list)):
                continue
            
            image_ids = [os.path.basename(image_name).split('.')[0][-7:] for image_name in image_file_list]
            depth_ids = [os.path.basename(depth_name).split('.')[0][-7:] for depth_name in depth_file_list]
            
            image_clip_list = []
            depth_clip_list = []
            extr_clip_list = []
            
            for image_id in image_ids:
                if image_id not in depth_ids:
                    continue
                image_clip_list.append(
                    os.path.join(scene, "undistort_rgb", "vignette{}.jpg".format(image_id)))
                depth_clip_list.append(
                    os.path.join(scene, "undistort_depth", "depth{}.png".format(image_id)))
                extr = extr_list[int(image_id)]
                extr = extr @ trans_device_from_camera
                extr = np.linalg.inv(extr)
                extr_clip_list.append(extr)
                if (len(image_clip_list) + 1) * self.samp_interval > self.clip_length:
                    sample_list.append(dict(
                        input_images=image_clip_list, 
                        input_depths=depth_clip_list, 
                        input_depth_masks=None, 
                        input_extrs=np.stack(extr_clip_list), 
                        input_intrs=np.stack([intr for _ in range(len(image_clip_list))]), 
                    ))
                    image_clip_list = []
                    depth_clip_list = []
                    extr_clip_list = []
            # if len(image_clip_list) > 0:
            #     sample_list.append(dict(
            #         input_images=image_clip_list, 
            #         input_depths=depth_clip_list, 
            #         input_depth_masks=None, 
            #         input_extrs=np.stack(extr_clip_list), 
            #         input_intrs=np.stack([intr for _ in range(len(image_clip_list))]), 
            #     ))
    
        return sample_list


if __name__ == '__main__':
    ase_root = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-251125/Reliev3R_data_sample/ASE_Sample"
    
    dataset = ASE_Dataset(sample_min=8, 
                          sample_max=8, 
                          clip_length=16, 
                          samp_interval=1, 
                          data_root=ase_root, 
                          match_eps=0.001)
    
    import os
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from functools import partial
    from dataset.utils.common import customized_collate_fn_placeholder, worker_init_fn
    from dataset.utils.vis import single_depth_intr_to_pcd, depth_maps_to_pcd, vis_batch_data
    
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=customized_collate_fn_placeholder, 
                            num_workers=2, 
                            persistent_workers=True, 
                            multiprocessing_context="spawn", 
                            worker_init_fn=partial(worker_init_fn, device_id=0))
    
    for idx, data in enumerate(dataloader):
        print("[ DEBUG ] data sample {} load success~".format(idx))
        print("          image shape: {}, image range: {}->{}".format(
            data["input_images"].shape, data["input_images"].min().item(), data["input_images"].max().item()))
        print("          depth shape: {}, depth range: {}->{}".format(
            data["input_depths"].shape, data["input_depths"].min().item(), data["input_depths"].max().item()))
        # print("          depth mask shape: {}, depth mask range: {}->{}".format(
        #     data["input_depth_masks"].shape, data["input_depth_masks"].min().item(), data["input_depth_masks"].max().item()))
        print("          tracking shape: {}, tracking range: {}->{}".format(
            data["input_tracks1"].shape, data["input_tracks1"].min().item(), data["input_tracks1"].max().item()))
        print("          extr shape: {}".format(data["input_viewmats"].shape))
        print("          intr shape: {}".format(data["input_Ks"].shape))
        # print("[ DEBUG ] extrs: {}".format(data["input_extrs"]))
        # print("[ DEBUG ] intrs: {}".format(data["input_intrs"]))
        B, V, K = data["input_visibs0"].shape
        print("[ DEBUG ] valid matching ratio: {}/{}".format(data["input_visibs0"].sum(), B * V * K))
        print("[ DEBUG ] valid keypoint range: {}->{}".format(
            data["input_tracks1"][data["input_visibs1"]].min(), 
            data["input_tracks1"][data["input_visibs1"]].max()
        ))

        # import time
        # time.sleep(2000)
        os.makedirs(os.path.join(ase_root, "vis_samp"), exist_ok=True)
        vis_img = vis_batch_data(data, keypoint_radius=1)
        torchvision.utils.save_image(vis_img, os.path.join(ase_root, "vis_samp", "vis_samp_{:03}.jpg".format(idx)))
        single_depth_intr_to_pcd(os.path.join(ase_root, "vis_samp", "vis_samp_{:03d}_single.ply".format(idx)), 
                                 data["input_images"][0, 0].permute(1, 2, 0), 
                                 data["input_depths"][0, 0].permute(1, 2, 0)[..., 0], 
                                 data["input_Ks"][0, 0])
        depth_maps_to_pcd(
            output_path=os.path.join(ase_root, "vis_samp", "vis_samp_{:03d}_all.ply".format(idx)), 
            images=data["input_images"][0].permute(0, 2, 3, 1), 
            depths=data["input_depths"][0].permute(0, 2, 3, 1)[..., 0], 
            extrs=torch.linalg.inv(data["input_viewmats"][0]), 
            intrs=data["input_Ks"][0], 
        )
    
    print("[ INFO ] Done. ")