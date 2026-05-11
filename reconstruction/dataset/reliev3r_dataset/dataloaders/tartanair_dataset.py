import os
from glob import glob
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_dataset import Synthetic_Dataset



def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    return np.array(SE)


class TartanAir_Dataset(Synthetic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_upperbound = 1000
    
    def create_sample_list(self):
        scene_list = sorted(glob(os.path.join(self.data_root, "*", "*", "*")))
        sample_list = []
        for scene_id, scene in enumerate(tqdm(scene_list)):
            # if scene_id > 0:
            #     break
            for side in ["left", "right"]:
                image_folder = os.path.join(scene, "image_{}".format(side))
                depth_folder = os.path.join(scene, "depth_{}".format(side))
                cam_file = os.path.join(scene, "pose_{}.txt".format(side))
                if not (os.path.exists(image_folder) and 
                        os.path.exists(depth_folder) and 
                        os.path.exists(cam_file)):
                    continue
                image_file_list = sorted(glob(os.path.join(image_folder, "*_{}.png".format(side))))
                depth_file_list = sorted(glob(os.path.join(depth_folder, "*_{}_depth.npy".format(side))))
                extr_meta = np.loadtxt(cam_file)
                intr = np.array([[0.5, 0, 0.5], [0, 0.66666666, 0.5], [0, 0, 1]]) # in opencv coord
                
                # assert len(image_file_list) == len(depth_file_list) and len(image_file_list) == len(extr_meta)
                
                image_clip_list = []
                depth_clip_list = []
                extr_clip_list = []
                for image_file, depth_file, pose_quat in \
                    zip(image_file_list[::self.samp_interval], 
                        depth_file_list[::self.samp_interval], 
                        extr_meta[::self.samp_interval]):
                    image_clip_list.append(image_file)
                    depth_clip_list.append(depth_file)
                    extr = pos_quat2SE(pose_quat) # in ned coord
                    trans = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
                    # extr = trans @ extr
                    extr = extr @ np.linalg.inv(trans)
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
    tartanair_root = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-251125/Reliev3R_data_sample/TartanAir_Sample"
    
    dataset = TartanAir_Dataset(sample_min=8, 
                                sample_max=8, 
                                clip_length=16, 
                                samp_interval=1, 
                                data_root=tartanair_root, 
                                depth_upperbound=1000, 
                                match_eps=0.001, 
                                train_proc=0.0)
    
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
        torchvision.utils.save_image(vis_img, os.path.join(tartanair_root, "vis_samp_{:03}.jpg".format(idx)))
        
        single_depth_intr_to_pcd(os.path.join(tartanair_root, "vis_samp_{:03d}_single.ply".format(idx)), 
                                 data["input_images"][0, 0].permute(1, 2, 0), 
                                 data["input_depths"][0, 0].permute(1, 2, 0)[..., 0], 
                                 data["input_intrs"][0, 0])
        depth_maps_to_pcd(
            output_path=os.path.join(tartanair_root, "vis_samp_{:03d}_all.ply".format(idx)), 
            images=data["input_images"][0].permute(0, 2, 3, 1), 
            depths=data["input_depths"][0].permute(0, 2, 3, 1)[..., 0], 
            extrs=torch.linalg.inv(data["input_extrs"][0]), 
            intrs=data["input_intrs"][0], 
        )
    
    print("[ INFO ] Done. ")