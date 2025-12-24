import math
from torch.optim.lr_scheduler import LambdaLR
def cosine_scheduler_with_warmup(optimizer, base_lr, warmup_steps, total_steps, min_lr=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr + (base_lr - min_lr) * cosine_decay) / base_lr
    return LambdaLR(optimizer, lr_lambda)


from torch.optim.lr_scheduler import _LRScheduler
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_epochs, base_lr, max_lr, T_max, last_epoch=-1):
        """
        初始化学习率调度器
        :param optimizer: 优化器
        :param warmup_steps: warmup阶段的epoch数
        :param max_epochs: 总的epoch数
        :param base_lr: 最小学习率
        :param max_lr: 最大学习率
        :param T_max: 余弦衰减的周期（即周期性余弦衰减的周期数）
        :param last_epoch: 上一个epoch的索引
        """
        self.warmup_epochs = warmup_steps  # warmup阶段的epoch数
        self.max_epochs = max_epochs  # 总的epoch数
        self.base_lr = base_lr  # 最小学习率
        self.max_lr = max_lr  # 最大学习率
        self.T_max = T_max  # 余弦衰减的周期
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        计算当前epoch的学习率
        :return: 当前epoch的学习率
        """
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # warmup阶段：学习率从 base_lr 线性增长到 max_lr
            lr = self.base_lr + (self.max_lr - self.base_lr) * epoch / self.warmup_epochs
        else:
            # cosine annealing 阶段：从 max_lr 衰减到 base_lr
            cos_inner = (epoch - self.warmup_epochs) % self.T_max
            # cos_out = 0.5 * (1 + torch.cos(torch.pi * cos_inner / self.T_max))
            cos_out = 0.5 * (1 + math.cos(math.pi * cos_inner / self.T_max))
            lr = self.base_lr + (self.max_lr - self.base_lr) * cos_out

        return [lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        state_dict = {
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'T_max': self.T_max,
            'last_epoch': self.last_epoch,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.max_epochs = state_dict['max_epochs']
        self.base_lr = state_dict['base_lr']
        self.max_lr = state_dict['max_lr']
        self.T_max = state_dict['T_max']
        self.last_epoch = state_dict['last_epoch']


class WarmupConstLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_epochs, base_lr, max_lr, last_epoch=-1):
        """
        初始化学习率调度器
        :param optimizer: 优化器
        :param warmup_steps: warmup阶段的epoch数
        :param max_epochs: 总的epoch数
        :param base_lr: 最小学习率
        :param max_lr: 最大学习率
        :param last_epoch: 上一个epoch的索引
        """
        self.warmup_epochs = warmup_steps  # warmup阶段的epoch数
        self.max_epochs = max_epochs  # 总的epoch数
        self.base_lr = base_lr  # 最小学习率
        self.max_lr = max_lr  # 最大学习率
        super(WarmupConstLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        计算当前epoch的学习率
        :return: 当前epoch的学习率
        """
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # warmup阶段：学习率从 base_lr 线性增长到 max_lr
            lr = self.base_lr + (self.max_lr - self.base_lr) * epoch / self.warmup_epochs
        else:
            lr = self.max_lr

        return [lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        state_dict = {
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'last_epoch': self.last_epoch,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.max_epochs = state_dict['max_epochs']
        self.base_lr = state_dict['base_lr']
        self.max_lr = state_dict['max_lr']
        self.last_epoch = state_dict['last_epoch']

