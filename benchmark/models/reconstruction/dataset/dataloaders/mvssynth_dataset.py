import os
from glob import glob
from tqdm import tqdm
import numpy as np
import json

from dataset.dataloaders.base_dataset import Synthetic_Dataset


class MVSSynth_Dataset(Synthetic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[ INFO ] {} samples found for MVS-Synth dataset as {}.".format(
            len(self), "training set" if self.is_train else "validation set"))
    
    def create_sample_list(self):
        scene_list = sorted(glob(os.path.join(self.data_root, "0*")))
        sample_list = []
        for scene_id, scene in enumerate(tqdm(scene_list)):
            # if scene_id > 0:
            #     break
            image_folder = os.path.join(scene, "images")
            depth_folder = os.path.join(scene, "depths")
            cam_folder = os.path.join(scene, "poses")
            image_file_list = sorted(glob(os.path.join(image_folder, "*.png")))
            depth_file_list = sorted(glob(os.path.join(depth_folder, "*.exr")))
            cam_file_list = sorted(glob(os.path.join(cam_folder, "*.json")))
            image_ids = [os.path.basename(image_file).split('.')[0] for image_file in image_file_list]
            
            image_clip_list = []
            depth_clip_list = []
            extr_clip_list = []
            intr_clip_list = []
            for image_id in image_ids[::self.samp_interval]:
                image_file = os.path.join(image_folder, "{}.png".format(image_id))
                depth_file = os.path.join(depth_folder, "{}.exr".format(image_id))
                cam_file = os.path.join(cam_folder, "{}.json".format(image_id))
                if not (os.path.exists(image_file) and os.path.exists(depth_file) and os.path.exists(cam_file)):
                    continue
                image_clip_list.append(image_file)
                depth_clip_list.append(depth_file)
                with open(cam_file, "r") as fp:
                    cam_info = json.load(fp)
                extr = np.array(cam_info["extrinsic"])
                extr_clip_list.append(extr)
                intr = np.eye(3)
                intr[0, 2] = 0.5
                intr[1, 2] = 0.5
                intr[0, 0] = 771.89826242 / (2 * 640)
                intr[1, 1] = cam_info["f_y"] / (2 * cam_info["c_y"])
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
            if len(image_clip_list) > 0:
                sample_list.append(dict(
                    input_images=image_clip_list, 
                    input_depths=depth_clip_list, 
                    input_depth_masks=None, 
                    input_extrs=np.stack(extr_clip_list), 
                    input_intrs=np.stack(intr_clip_list), 
                ))
        
        return sample_list


if __name__ == '__main__':
    mvssynth_root = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-251125/Reliev3R_data_sample/MVS-Synth_Sample/GTAV_540"
    
    dataset = MVSSynth_Dataset(sample_min=8, 
                               sample_max=8, 
                               clip_length=16, 
                               samp_interval=1, 
                               data_root=mvssynth_root, 
                               match_eps=0.1)
    
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
        torchvision.utils.save_image(vis_img, os.path.join(mvssynth_root, "vis_samp_{:03}.jpg".format(idx)))
        
        single_depth_intr_to_pcd(os.path.join(mvssynth_root, "vis_samp_{:03d}_single.ply".format(idx)), 
                                 data["input_images"][0, 0].permute(1, 2, 0), 
                                 data["input_depths"][0, 0].permute(1, 2, 0)[..., 0], 
                                 data["input_intrs"][0, 0])
        depth_maps_to_pcd(
            output_path=os.path.join(mvssynth_root, "vis_samp_{:03d}_all.ply".format(idx)), 
            images=data["input_images"][0].permute(0, 2, 3, 1), 
            depths=data["input_depths"][0].permute(0, 2, 3, 1)[..., 0], 
            extrs=torch.linalg.inv(data["input_extrs"][0]), 
            intrs=data["input_intrs"][0], 
        )
    
    print("[ INFO ] Done. ")