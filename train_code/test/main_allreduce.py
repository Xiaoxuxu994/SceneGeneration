import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from multiprocessing import Queue, Process
import time
from tqdm import tqdm
import random

# --------------------------
# 1. 定义随机数据加载器（仅示例）
# --------------------------Q
def get_data_loader(rank, batch_size=32):
    """为每个GPU生成不同的随机数据"""
    torch.manual_seed(42 + rank)  # 确保各进程数据不同
    data = torch.randn(1000, 10)  # 1000个样本，特征维度10
    labels = torch.randint(0, 2, (1000, 1)).float()  # 二分类标签
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_val_loader(batch_size=32):
    """CPU验证进程的随机数据"""
    val_data = torch.randn(200, 10)
    val_labels = torch.randint(0, 2, (200, 1)).float()
    val_dataset = TensorDataset(val_data, val_labels)
    return DataLoader(val_dataset, batch_size=batch_size)

# --------------------------
# 2. 定义简易模型
# --------------------------
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# --------------------------
# 3. GPU工作进程（DDP训练）
# --------------------------
def gpu_worker(rank, world_size, shared_queue, validation_signal):
    # 初始化分布式
    dist.init_process_group(
        backend="hccl",
        init_method="tcp://localhost:12345",
        rank=rank,
        world_size=world_size
    )
    torch.npu.set_device(rank)
    # 模型和数据
    model = MyModel().to(f'npu:{rank}')
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    train_loader = get_data_loader(rank)
    
    
    
    # 模拟训练循环
    for epoch in tqdm(range(10)):  # 2个epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(f'npu:{rank}'), target.to(f'npu:{rank}')
            
            # 前向+反向
            output = ddp_model(data)
            loss = nn.BCELoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            
            # 异步发送梯度到CPU进程（模拟）
            if rank == 0:  # 仅主GPU发送
                grads = [p.grad.cpu() for p in ddp_model.parameters()]
                shared_queue.put(("grads", grads))
                print(f"Rank {rank} sent gradients to CPU")
            
            # 非阻塞检查验证信号
            if not validation_signal.empty():
                val_result = validation_signal.get()
                print(f"Rank {rank} received val result: {val_result}")
            
            optimizer.step()
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

# --------------------------
# 4. CPU验证进程
# --------------------------
def cpu_worker(shared_queue, validation_signal):
    val_model = MyModel()  # CPU模型
    val_loader = get_val_loader()
    
    while True:
        # 接收梯度（非阻塞）
        try:
            msg, grads = shared_queue.get(timeout=1.0)
            if msg == "grads":
                print("CPU received gradients")
                
                # 模拟验证（随机延迟）
                time.sleep(random.uniform(0.1, 0.5))
                val_loss = np.random.rand()  # 随机验证损失
                
                # 发送验证结果回GPU
                validation_signal.put(("val_loss", val_loss))
        except:
            continue

def sync_worker(rank, world_size):
    # 初始化分布式
    dist.init_process_group(
        backend="hccl",
        init_method="tcp://localhost:12345",
        rank=rank,
        world_size=world_size
    )
    torch.npu.set_device(rank)
    # 模型和数据
    model = MyModel().to(f'npu:{rank}')
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    train_loader = get_data_loader(rank)

    
    # 模拟训练循环
    for epoch in tqdm(range(10)):  # 2个epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(f'npu:{rank}'), target.to(f'npu:{rank}')
            
            # 前向+反向
            output = ddp_model(data)
            loss = nn.BCELoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            
            # 异步发送梯度到CPU进程（模拟）
            # if rank == 0:  # 仅主GPU发送
            # grads = [p.grad.cpu() for p in ddp_model.parameters()]
            # shared_queue.put(("grads", grads))
            # print(f"Rank {rank} sent gradients to CPU")
            # print("CPU received gradients")
            
            # 模拟验证（随机延迟）
            time.sleep(random.uniform(0.1, 0.5))
            val_loss = np.random.rand()  # 随机验证损失
            
            # 非阻塞检查验证信号
            # if not validation_signal.empty():
            #     val_result = validation_signal.get()
            #     print(f"Rank {rank} received val result: {val_result}")
            
            optimizer.step()
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

# --------------------------
# 5. 启动脚本
# --------------------------
if __name__ == "__main__":
    # 共享通信组件
    # Async test
    manager = mp.Manager()
    shared_queue = manager.Queue()
    validation_signal = manager.Queue()
    
    # 启动CPU验证进程
    cpu_proc = Process(
        target=cpu_worker,
        args=(shared_queue, validation_signal),
        daemon=True
    )
    cpu_proc.start()
    
    # 启动GPU进程（模拟2卡）
    world_size = 2
    mp.spawn(
        gpu_worker,
        args=(world_size, shared_queue, validation_signal),
        nprocs=world_size,
        join=True
    )
    
    cpu_proc.terminate()  # 安全终止
    # SyncTest
    # world_size = 2
    # mp.spawn(
    #     sync_worker,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True
    # )