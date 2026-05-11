import numpy as np

from .reliev3r_dataset.dataloaders import create_mixed_dataloader as weaksup_dataloader, load_data_to_device
from .pi3_dataset import create_dataloader as fullsup_dataloader


DATALOADER_FUNC = {
    "pi3":      fullsup_dataloader, 
    "reliev3r": weaksup_dataloader, 
}


# class Reliev3R_MixDataLoader:
    
#     def __init__(self, cfg, mode, reliev3r_batch_size):
#         self.dataset_weights = cfg.weights
#         self.cfg = cfg
#         total_weights = 0
#         for k, v in self.dataset_weights.items():
#             total_weights += v
#         for k in self.dataset_weights.keys():
#             self.dataset_weights[k] /= total_weights
#         self.mode = mode
#         self.dataloaders = dict()
#         self.dataloader_iters = dict()
#         self.pi3_epoch_count = 0
#         for dataloader_name in DATALOADER_FUNC.keys():
#             self.dataloaders[dataloader_name] = DATALOADER_FUNC[dataloader_name](cfg=cfg.dataloaders[dataloader_name], 
#                                                                                  mode=mode, batch_size=reliev3r_batch_size)
#         self.set_epoch(self.pi3_epoch_count)
#         for dataloader_name in DATALOADER_FUNC.keys():
#             self.dataloader_iters[dataloader_name] = iter(self.dataloaders[dataloader_name])
        
#         self.rng = np.random.default_rng(seed=888)
    
#     def set_epoch(self, epoch=0):
#         # ---------- train ----------
#         if hasattr(self.dataloaders["pi3"], 'dataset') and hasattr(self.dataloaders["pi3"].dataset, 'set_epoch'):
#             # custom dataset, can pass base_seed
#             try:
#                 self.dataloaders["pi3"].dataset.set_epoch(epoch, base_seed=self.cfg.dataloaders.pi3.train.base_seed)
#             except TypeError:
#                 self.dataloaders["pi3"].dataset.set_epoch(epoch)
        
#         if hasattr(self.dataloaders["pi3"], 'sampler') and hasattr(self.dataloaders["pi3"].sampler, 'set_epoch'):
#             # PyTorch DistributedSampler 只能传 epoch
#             self.dataloaders["pi3"].sampler.set_epoch(epoch)

#         # handle accelerate wrapped dataloader
#         if (
#             hasattr(self.dataloaders["pi3"], 'batch_sampler')
#             and hasattr(self.dataloaders["pi3"].batch_sampler, 'batch_sampler')
#             and hasattr(self.dataloaders["pi3"].batch_sampler.batch_sampler, 'sampler')
#             and hasattr(self.dataloaders["pi3"].batch_sampler.batch_sampler.sampler, 'set_epoch')
#         ):
#             self.dataloaders["pi3"].batch_sampler.batch_sampler.sampler.set_epoch(epoch)

#         if hasattr(self.dataloaders["pi3"], 'batch_sampler') and hasattr(self.dataloaders["pi3"].batch_sampler, 'set_epoch'):
#             try:
#                 self.dataloaders["pi3"].batch_sampler.set_epoch(epoch, base_seed=self.cfg.dataloaders.pi3.train.base_seed)
#             except TypeError:
#                 self.dataloaders["pi3"].batch_sampler.set_epoch(epoch)
    
#     def sample(self, step):
#         dataset_names = []
#         dataset_weights = []
#         for k, v in self.dataset_weights.items():
#             dataset_names.append(k)
#             dataset_weights.append(v)
#         dataset_idx = self.rng.choice(np.arange(len(dataset_names)), size=1, p=np.array(dataset_weights))[0]
#         this_step_dataset_name = dataset_names[dataset_idx]
        
#         data_iter = self.dataloader_iters[this_step_dataset_name]
#         while True:
#             try:
#                 data = next(data_iter)
#                 if this_step_dataset_name == "reliev3r":
#                     if data is not None:
#                         data = load_data_to_device(data)
#                         break
#                     print("[ WARNING ] Error data found at step {} with reliev3r dataloader!".format(step))
#                 elif this_step_dataset_name == "pi3":
#                     data = load_data_to_device(data)
#                     break
#                     print("[ WARNING ] Error data found at step {} with pi3 dataloader!".format(step))
#                 else:
#                     raise NotImplementedError("Dataset class {} is not implemented.".format(this_step_dataset_name))
#             except StopIteration:
#                 if this_step_dataset_name == "pi3":
#                     self.pi3_epoch_count += 1
#                     self.set_epoch(self.pi3_epoch_count)
#                 data_iter = iter(self.dataloaders[this_step_dataset_name])
#         self.dataloader_iters[this_step_dataset_name] = data_iter
        
#         return data


class Reliev3R_MixDataLoader:
    
    def __init__(self, cfg, mode, reliev3r_batch_size):
        
        self.cfg = cfg
        self.mode = mode

        self.dataset_weights = dict(cfg.weights)
        # total_weights = 0
        # for k, v in self.dataset_weights.items():
        #     total_weights += v
        # for k in self.dataset_weights.keys():
        #     self.dataset_weights[k] /= total_weights

        self.dataloaders = dict()
        self.dataloader_iters = dict()
        self.pi3_epoch_count = 0
        for dataloader_name in DATALOADER_FUNC.keys():
            self.dataloaders[dataloader_name] = DATALOADER_FUNC[dataloader_name](cfg=cfg.dataloaders[dataloader_name], 
                                                                                 mode=mode, batch_size=reliev3r_batch_size)
        self.set_epoch(self.pi3_epoch_count)
        for dataloader_name in DATALOADER_FUNC.keys():
            self.dataloader_iters[dataloader_name] = iter(self.dataloaders[dataloader_name])
        
        self.rng = np.random.default_rng(seed=888)

        self.mix_start_step = getattr(cfg, "mix_start_step", 1000)
        # 预计算两个阶段的采样分布
        self.stage1_dataset_names, self.stage1_dataset_weights = self._build_stage1_distribution()
        self.stage2_dataset_names, self.stage2_dataset_weights = self._build_stage2_distribution()

        print(f"[ INFO ] mix_start_step = {self.mix_start_step}")
        print(f"[ INFO ] stage1 datasets = {self.stage1_dataset_names}, weights = {self.stage1_dataset_weights}")
        print(f"[ INFO ] stage2 datasets = {self.stage2_dataset_names}, weights = {self.stage2_dataset_weights}")
    
    def _normalize_weights(self, names, weights):
        if len(names) == 0:
            raise RuntimeError("No valid datasets to sample from.")
        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise RuntimeError(f"Invalid weights: {weights}")
        weights = weights / weight_sum
        return names, weights
    
    def _build_stage1_distribution(self):
        """
        stage1: step < mix_start_step
        只允许使用 pi3
        且 pi3 weight 必须 > 0
        """
        w_pi3 = float(self.dataset_weights.get("pi3", 0.0))
        if w_pi3 <= 0:
            raise RuntimeError(
                f"Warmup stage requires pi3, but cfg.weights['pi3']={w_pi3}. "
                f"Please set pi3 weight > 0."
            )
        return ["pi3"], np.array([1.0], dtype=np.float64)

    def _build_stage2_distribution(self):
        """
        stage2: step >= mix_start_step
        允许使用 pi3 + reliev3r
        其中 weight <= 0 的数据集会被过滤掉
        """
        names = []
        weights = []

        for name in ["pi3", "reliev3r"]:
            w = float(self.dataset_weights.get(name, 0.0))
            if w > 0:
                names.append(name)
                weights.append(w)

        return self._normalize_weights(names, weights)
    
    def set_epoch(self, epoch=0):
        # ---------- train ----------
        if hasattr(self.dataloaders["pi3"], 'dataset') and hasattr(self.dataloaders["pi3"].dataset, 'set_epoch'):
            # custom dataset, can pass base_seed
            try:
                self.dataloaders["pi3"].dataset.set_epoch(epoch, base_seed=self.cfg.dataloaders.pi3.train.base_seed)
            except TypeError:
                self.dataloaders["pi3"].dataset.set_epoch(epoch)
        
        if hasattr(self.dataloaders["pi3"], 'sampler') and hasattr(self.dataloaders["pi3"].sampler, 'set_epoch'):
            # PyTorch DistributedSampler 只能传 epoch
            self.dataloaders["pi3"].sampler.set_epoch(epoch)

        # handle accelerate wrapped dataloader
        if (
            hasattr(self.dataloaders["pi3"], 'batch_sampler')
            and hasattr(self.dataloaders["pi3"].batch_sampler, 'batch_sampler')
            and hasattr(self.dataloaders["pi3"].batch_sampler.batch_sampler, 'sampler')
            and hasattr(self.dataloaders["pi3"].batch_sampler.batch_sampler.sampler, 'set_epoch')
        ):
            self.dataloaders["pi3"].batch_sampler.batch_sampler.sampler.set_epoch(epoch)

        if hasattr(self.dataloaders["pi3"], 'batch_sampler') and hasattr(self.dataloaders["pi3"].batch_sampler, 'set_epoch'):
            try:
                self.dataloaders["pi3"].batch_sampler.set_epoch(epoch, base_seed=self.cfg.dataloaders.pi3.train.base_seed)
            except TypeError:
                self.dataloaders["pi3"].batch_sampler.set_epoch(epoch)
    
    def _choose_dataset_name(self, step):
        if step < self.mix_start_step:
            dataset_names = self.stage1_dataset_names
            dataset_weights = self.stage1_dataset_weights
        else:
            dataset_names = self.stage2_dataset_names
            dataset_weights = self.stage2_dataset_weights

        dataset_idx = self.rng.choice(
            len(dataset_names),
            p=dataset_weights
        )
        return dataset_names[dataset_idx]

    def sample(self, step):
        this_step_dataset_name = self._choose_dataset_name(step)
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
                if this_step_dataset_name == "pi3":
                    self.pi3_epoch_count += 1
                    self.set_epoch(self.pi3_epoch_count)
                data_iter = iter(self.dataloaders[this_step_dataset_name])
        self.dataloader_iters[this_step_dataset_name] = data_iter
        
        return data