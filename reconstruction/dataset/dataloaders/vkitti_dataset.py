import os
from glob import glob
from tqdm import tqdm
import numpy as np

from dataset.dataloaders.base_dataset import Synthetic_Dataset


class VKitti_Dataset(Synthetic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_scale = 0.01
        self.depth_upperbound = 655.0
    
    def create_sample_list(self):
        scene_names = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        weather_names = [
            "15-deg-left", 
            "15-deg-right", 
            "30-deg-left", 
            "30-deg-right", 
            "clone", 
            "fog", 
            "morning", 
            "overcast", 
            "rain", 
            "sunset", 
        ]
        scene_list = []
        sample_list = []
        for scene_name in scene_names:
            for weather_name in weather_names:
                scene = os.path.join(self.data_root, scene_name, weather_name)
                if os.path.exists(scene):
                    scene_list.append(scene)
        for scene_id, scene in enumerate(tqdm(scene_list)):
            # if scene_id > 0:
            #     break
            extr_file = os.path.join(scene, "extrinsic.txt")
            intr_file = os.path.join(scene, "intrinsic.txt")
            if not (os.path.exists(extr_file) and os.path.exists(intr_file)):
                continue
            for scene_cam_id in range(2):
                with open(extr_file, "r") as fp:
                    lines = fp.readlines()
                    extr_list = [np.array([float(ele) for ele in line.split(' ')[2:]]).reshape(4, 4) for line in lines[(1 + scene_cam_id)::2]]
                with open(intr_file, "r") as fp:
                    lines = fp.readlines()
                    intr_list = []
                    for line in lines[(1 + scene_cam_id)::2]:
                        eles = line.split(' ')
                        intr = np.eye(3)
                        intr[0, 0] = float(eles[-4]) / 1242
                        intr[1, 1] = float(eles[-3]) / 375
                        intr[0, 2] = 0.5
                        intr[1, 2] = 0.5
                        intr_list.append(intr)
                image_file_list = sorted(glob(os.path.join(scene, "frames", "rgb", "Camera_{}".format(scene_cam_id), "rgb_*.jpg")))
                depth_file_list = sorted(glob(os.path.join(scene, "frames", "depth", "Camera_{}".format(scene_cam_id), "depth_*.png")))
                
                # assert len(image_file_list) == len(depth_file_list)
                # assert (len(image_file_list) == len(depth_file_list) and 
                #         len(image_file_list) == len(extr_list) and 
                #         len(image_file_list) == len(intr_list))
                
                image_clip_list = []
                depth_clip_list = []
                extr_clip_list = []
                intr_clip_list = []
                for image_file, depth_file, extr, intr in \
                    zip(image_file_list[::self.samp_interval], 
                        depth_file_list[::self.samp_interval], 
                        extr_list[::self.samp_interval], 
                        intr_list[::self.samp_interval]):
                    image_clip_list.append(image_file)
                    depth_clip_list.append(depth_file)
                    extr_clip_list.append(extr)
                    intr_clip_list.append(intr)
                    if (len(image_clip_list) + 1) * self.samp_interval > self.clip_length:
                        sample_list.append(dict(
                            input_images=image_clip_list, 
                            input_depths=depth_clip_list, 
                            input_depth_masks=None, 
                            input_extrs=np.stack(extr_clip_list), 
                            input_intrs=np.stack(intr_clip_list), 
                        ))
                        image_clip_list = []
                        depth_clip_list = []
                        extr_clip_list = []
                        intr_clip_list = []
                # if len(image_clip_list) > 0:
                #     sample_list.append(dict(
                #         input_images=image_clip_list, 
                #         input_depths=depth_clip_list, 
                #         input_depth_masks=None, 
                #         input_extrs=np.stack(extr_clip_list), 
                #         input_intrs=np.stack(intr_clip_list), 
                #     ))
                    
        return sample_list


if __name__ == '__main__':
    vkitti_root = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-251125/Reliev3R_data_sample/vKitti2_Sample"
    
    dataset = VKitti_Dataset(sample_min=4, 
                             sample_max=4, 
                             clip_length=16, 
                             samp_interval=1, 
                             data_root=vkitti_root, 
                             match_eps=0.01)
    
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
        print("[ DEBUG ] data sample {} load success~".format(idx))
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
        vis_img = vis_batch_data(data, keypoint_radius=1)
        torchvision.utils.save_image(vis_img, os.path.join(vkitti_root, "vis_samp_{:03}.jpg".format(idx)))
        
        single_depth_intr_to_pcd(os.path.join(vkitti_root, "vis_samp_{:03d}_single.ply".format(idx)), 
                                 data["input_images"][0, 0].permute(1, 2, 0), 
                                 data["input_depths"][0, 0].permute(1, 2, 0)[..., 0], 
                                 data["input_intrs"][0, 0])
        depth_maps_to_pcd(
            output_path=os.path.join(vkitti_root, "vis_samp_{:03d}_all.ply".format(idx)), 
            images=data["input_images"][0].permute(0, 2, 3, 1), 
            depths=data["input_depths"][0].permute(0, 2, 3, 1)[..., 0], 
            extrs=torch.linalg.inv(data["input_extrs"][0]), 
            intrs=data["input_intrs"][0], 
        )
    
    print("[ INFO ] Done. ")