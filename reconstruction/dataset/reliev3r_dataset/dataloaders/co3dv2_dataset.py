import os
from .base_dataset import Realistic_Dataset


class CO3Dv2_Dataset(Realistic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if __name__ == "__main__":
    co3dv2_root = "/mnt/diskb/chenyouyu/workspace/code/Reliev3R-251125/Reliev3R_data_sample/co3dv2_Sample"
    dataset = CO3Dv2_Dataset(sample_min=8, 
                             sample_max=8, 
                             data_root=co3dv2_root, 
                             image_format=".jpg")
    
    import os
    import torchvision
    from torch.utils.data import DataLoader
    from functools import partial
    from dataset.utils.common import customized_collate_fn, worker_init_fn
    from dataset.utils.vis import vis_batch_data
    
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            collate_fn=customized_collate_fn, 
                            num_workers=2, 
                            persistent_workers=True, 
                            multiprocessing_context="spawn", 
                            worker_init_fn=partial(worker_init_fn, device_id=0))
    
    for idx, data in enumerate(dataloader):
        # print("[ DEBUG ] load success~")
        # print("          image shape: {}, image range: {}->{}".format(
        #     data["input_images"].shape, data["input_images"].min().item(), data["input_images"].max().item()))
        # print("          depth shape: {}, depth range: {}->{}".format(
        #     data["input_depths"].shape, data["input_depths"].min().item(), data["input_depths"].max().item()))
        # print("          depth mask shape: {}, depth mask range: {}->{}".format(
        #     data["input_depth_masks"].shape, data["input_depth_masks"].min().item(), data["input_depth_masks"].max().item()))
        # print("          tracking shape: {}, tracking range: {}->{}".format(
        #     data["input_tracks1"].shape, data["input_tracks1"].min().item(), data["input_tracks1"].max().item()))
        # print("          extr shape: {}".format(data["input_extrs"].shape))
        # print("          intr shape: {}".format(data["input_intrs"].shape))
        vis_img = vis_batch_data(data)
        torchvision.utils.save_image(vis_img, os.path.join(co3dv2_root, "vis_samp_{:03}.jpg".format(idx)))
    
    print("[ INFO ] Done. ")