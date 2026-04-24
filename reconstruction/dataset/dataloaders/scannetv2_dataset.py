import os
from glob import glob
from tqdm import tqdm
import numpy as np

from dataset.dataloaders.base_dataset import Synthetic_Dataset


class ScanNetv2_Dataset(Synthetic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_scale = 0.001
        self.depth_upperbound = 655.0
    
    def create_sample_list(self):
        scene_list = sorted(glob(os.path.join(self.data_root, "scene*")))
        sample_list = []
        for scene in tqdm(scene_list):
            view_info = np.load(os.path.join(scene, "viewInfo.npz"), allow_pickle=True)
            image_file_names = view_info["image_filenames"]
            image_file_list = [os.path.join(scene, "color", f) for f in image_file_names]
            depth_file_list = [os.path.join(scene, "depth", f.split('.')[0] + ".png") for f in image_file_names]
            extrs_list = view_info["T_w2c"]
            intrs_list = view_info["K_norm"]
            
            image_clip_list = []
            depth_clip_list = []
            extr_clip_list = []
            intr_clip_list = []
            for image_file, depth_file, extr, intr in zip(
                image_file_list[::self.samp_interval], depth_file_list[::self.samp_interval], 
                extrs_list[::self.samp_interval], intrs_list[::self.samp_interval]
            ):
                image_clip_list.append(image_file)
                depth_clip_list.append(depth_file)
                extr_clip_list.append(extr)
                intr_clip_list.append(intr)
                assert os.path.exists(image_file) and os.path.exists(depth_file)
                
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
        
        return sample_list