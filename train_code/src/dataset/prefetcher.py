
import torch
import torch_npu
import torch.distributed as dist

class AsyncSamplerProfiler:
    def __init__(self, active_every=20):
        self.timings = []
        self.step = 0
        self.active_every = active_every

    def record(self, sampling_fn, stream=None):
        self.step += 1
        if self.step % self.active_every != 0:
            with torch_npu.npu.stream(stream or torch_npu.npu.current_stream()):
                sampling_fn()
            return None
        
        stream = stream or torch_npu.npu.current_stream()
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)

        with torch_npu.npu.stream(stream):
            start.record()
            sampling_fn()
            end.record()

        torch_npu.npu.current_stream().wait_stream(stream)
        end.synchronize()

        elapsed = start.elapsed_time(end)
        self.timings.append(elapsed)
        self.step = 0
        return elapsed

    def report(self, name="npu profile", last_n=20):
        if not self.timings:
            return
        recent = self.timings[-last_n:]
        avg = sum(recent) / len(recent)
        if dist.get_rank() == 0:
            print(f"[{name} Profiler] recent {len(recent)} avg cost: {avg:.2f} ms")

class Prefetcher:
    def __init__(self, dataloader, device, profiler = None, policy = None, sample_num=2048, random_start=True):
        self.dataloader = dataloader
        self.device = device
        self.load_iter = iter(self.dataloader)
        self.stream = torch_npu.npu.Stream()
        self.sample_num = sample_num
        self.random_start = random_start
        self.profiler = profiler
        self.policy = policy
        self.__init_buffers()
    
    def __init_buffers(self):
        self.next_batch = None
        self.preload()
    
    
    def fps_npu(self, points):
        with torch.no_grad():
            sampled_indices = torch.zeros((self.sample_num), dtype=torch.long, device=points.device)
            start = torch.randint(0, len(points), (1,), dtype=torch.long, device=points.device) if self.random_start else torch.tensor([0], dtype=torch.long, device=points.device)
            distances = torch.full((len(points),), 1e10, device=points.device)
            for i in range(self.sample_num):
                sampled_indices[i] = start 
                last_sampled = points[start]
                dist_tmp = torch.sum((points - last_sampled) ** 2, dim=1)
                distances = torch.minimum(distances, dist_tmp)
                start = torch.argmax(distances).to(torch.long)
        return sampled_indices

    def preload(self):
        try:
            batch = next(self.load_iter)
        except StopIteration:
            self.next_batch = None
            return
        
        batch = {
            k: v[0].to(device=self.device) #non_blocking=True) # only support batchsize = 1
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        if batch["fps_index"] is None:
            def sampling():
                batch["fps_index"] = self.fps_npu(batch["coord"])
            if self.profiler:
                duration = self.profiler.record(sampling, stream=self.stream)
                if self.policy:
                    self.policy.update(batch["coord"].shape[0], duration, device='npu')
            else:
                with torch_npu.npu.stream(self.stream):
                    batch["fps_index"] = self.fps_npu(batch["coord"])

        with torch_npu.npu.stream(self.stream):
            batch["fps_index"] = self.fps_npu(batch["coord"])
        
        self.next_batch = batch

    def next(self):
        torch_npu.npu.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch
    
    def reset(self, epoch):
        self.dataloader.sampler.set_epoch(epoch)
        self.load_iter = iter(self.dataloader)
        self.__init_buffers()