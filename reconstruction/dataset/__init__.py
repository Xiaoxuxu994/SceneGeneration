import numpy as np
import torch.distributed as dist

from .reliev3r_dataset.dataloaders import create_mixed_dataloader as weaksup_dataloader, load_data_to_device
from .pi3_dataset import create_dataloader as fullsup_dataloader


DATALOADER_FUNC = {
    "pi3":      fullsup_dataloader, 
    "reliev3r": weaksup_dataloader, 
}


class Reliev3R_MixDataLoader:
    
    def __init__(self, cfg, mode, reliev3r_batch_size):
        self.dataset_weights = cfg.weights
        self.cfg = cfg
        total_weights = 0
        for k, v in self.dataset_weights.items():
            total_weights += v
        for k in self.dataset_weights.keys():
            self.dataset_weights[k] /= total_weights
        self.mode = mode
        self.dataloaders = dict()
        self.dataloader_iters = dict()
        self.pi3_epoch_count = 0
        for dataloader_name in DATALOADER_FUNC.keys():
            # if self.dataset_weights[dataloader_name] > 0:
            self.dataloaders[dataloader_name] = DATALOADER_FUNC[dataloader_name](cfg=cfg.dataloaders[dataloader_name], 
                                                                                mode=mode, batch_size=reliev3r_batch_size)
        self.set_epoch(self.pi3_epoch_count)
        for dataloader_name in DATALOADER_FUNC.keys():
            # if self.dataset_weights[dataloader_name] > 0:
            self.dataloader_iters[dataloader_name] = iter(self.dataloaders[dataloader_name])
        
        self.rng = np.random.default_rng(seed=888)
    
    def set_epoch(self, epoch=0):
        # ---------- train ----------
        if hasattr(self.dataloaders["pi3"], 'dataset') and hasattr(self.dataloaders["pi3"].dataset, 'set_epoch'):
            # custom dataset, can pass base_seed
            try:
                self.dataloaders["pi3"].dataset.set_epoch(epoch, base_seed=self.cfg.dataloaders.pi3.train.base_seed)
                print("[ INFO ] dataset epoch set as {}".format(epoch))
            except TypeError:
                self.dataloaders["pi3"].dataset.set_epoch(epoch)
                print("[ INFO ] dataset epoch set as {}".format(epoch))
        
        if hasattr(self.dataloaders["pi3"], 'sampler') and hasattr(self.dataloaders["pi3"].sampler, 'set_epoch'):
            # PyTorch DistributedSampler 只能传 epoch
            self.dataloaders["pi3"].sampler.set_epoch(epoch)
            print("[ INFO ] sampler epoch set as {}".format(epoch))

        # handle accelerate wrapped dataloader
        if (
            hasattr(self.dataloaders["pi3"], 'batch_sampler')
            and hasattr(self.dataloaders["pi3"].batch_sampler, 'batch_sampler')
            and hasattr(self.dataloaders["pi3"].batch_sampler.batch_sampler, 'sampler')
            and hasattr(self.dataloaders["pi3"].batch_sampler.batch_sampler.sampler, 'set_epoch')
        ):
            self.dataloaders["pi3"].batch_sampler.batch_sampler.sampler.set_epoch(epoch)
            print("[ INFO ] batch sampler epoch set as {}".format(epoch))

        if hasattr(self.dataloaders["pi3"], 'batch_sampler') and hasattr(self.dataloaders["pi3"].batch_sampler, 'set_epoch'):
            try:
                self.dataloaders["pi3"].batch_sampler.set_epoch(epoch, base_seed=self.cfg.dataloaders.pi3.train.base_seed)
            except TypeError:
                self.dataloaders["pi3"].batch_sampler.set_epoch(epoch)
            print("[ INFO ] batch sampler epoch set as {}".format(epoch))
    
    def sample(self, step):
        dataset_names = []
        dataset_weights = []
        for k, v in self.dataset_weights.items():
            dataset_names.append(k)
            dataset_weights.append(v)
        dataset_idx = self.rng.choice(np.arange(len(dataset_names)), size=1, p=np.array(dataset_weights))[0]
        this_step_dataset_name = dataset_names[dataset_idx]
        
        data_iter = self.dataloader_iters[this_step_dataset_name]
        while True:
            try:
                data = next(data_iter)
                if this_step_dataset_name == "reliev3r":
                    if data is not None:
                        data = load_data_to_device(data)
                        break
                    print("[ WARNING ] Error data found at step {} with reliev3r dataloader!".format(step))
                elif this_step_dataset_name == "pi3":
                    data = load_data_to_device(data)
                    break
                    print("[ WARNING ] Error data found at step {} with pi3 dataloader!".format(step))
                else:
                    raise NotImplementedError("Dataset class {} is not implemented.".format(this_step_dataset_name))
            except StopIteration:
                # if this_step_dataset_name == "pi3":
                #     self.pi3_epoch_count += 1
                #     self.set_epoch(self.pi3_epoch_count)
                data_iter = iter(self.dataloaders[this_step_dataset_name])
        self.dataloader_iters[this_step_dataset_name] = data_iter
        
        return data
