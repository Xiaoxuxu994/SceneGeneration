"""NPU (Ascend 910B) check for the loop-transformer branch: correctness, peak memory and step time.

  python tools/loop_transformer/check_loop_model_npu.py bench           # 1 NPU: baseline vs loop runs fwd+bwd at the training shape
  torchrun --nproc_per_node 2 tools/loop_transformer/check_loop_model_npu.py ddp   # HCCL DDP with shared params + non-reentrant ckpt
Random init (no DINOv3 download); bf16 autocast like accelerate mixed_precision=bf16.
"""

import os
import sys
import time

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401  (maps torch.cuda.* calls in the model code)

sys.path.insert(0, os.environ.get("REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))))
import pi3.models.pi3_training as P  # noqa: E402

_orig_cfg = P.Pi3._build_backbone_cfg


def _cfg_no_download():
    cfg = _orig_cfg()
    b = cfg["dinov3"]["small"]["builder"]
    cfg["dinov3"]["small"]["builder"] = lambda pretrained=True, model_dir=None: b(pretrained=False)
    return cfg


P.Pi3._build_backbone_cfg = staticmethod(_cfg_no_download)
from pi3.models.pi3_loop import Pi3Loop  # noqa: E402

KW = dict(pos_type="rope100", model_size=os.environ.get("MODEL_SIZE", "dinov3_small_w384"), load_vggt=False, freeze_encoder=False,
          use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=0, ckpt=None)


def step(m, imgs, opt):
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        out = m(imgs)
    loss = out["points"].float().abs().mean() + out["camera_poses"].float().abs().mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss


def bench():
    dev = torch.device("npu:0")
    shapes = [(32, 8, 432, 592)]  # training recipe: 32 samples x 8 views per NPU at the largest resolution
    if os.environ.get("SHAPES"):  # e.g. SHAPES="16,8,432,592;24,8,432,592"  (batch, views, H, W)
        shapes = [tuple(int(v) for v in s.split(",")) for s in os.environ["SHAPES"].split(";")]
    no_ckpt = os.environ.get("NO_DEC_CKPT") == "1"  # decoder blocks without gradient checkpointing
    kwA = dict(KW, num_dec_blk_not_to_checkpoint=36 if no_ckpt else 0)
    # LOOPS="L,k;L,k" (loop body pairs, fixed K = worst case); default: the four exp1022 loop runs at K = k_max
    loops = [tuple(int(v) for v in s.split(",")) for s in os.environ.get("LOOPS", "1,18;2,9;6,3").split(";")]
    runs = [("base36" + ("_nockpt" if no_ckpt else ""), lambda: P.Pi3(**kwA))]
    for (L, K) in loops:
        runs.append((f"loop_L{L}_K{K}" + ("_nockpt" if no_ckpt else ""),
                     lambda L=L, K=K: Pi3Loop(**KW, loop_layers=L, loop_k_min=K, loop_k_max=K,
                                              loop_grad_checkpoint=not no_ckpt)))
    for name, ctor in runs:
        torch.manual_seed(0)
        m = ctor().to(dev).train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        npar = sum(p.numel() for p in m.parameters()) / 1e6
        for (b, n, h, w) in shapes:
            imgs = torch.rand(b, n, 3, h, w, device=dev)
            torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()
            try:
                step(m, imgs, opt)  # warmup
                torch.npu.synchronize(); t0 = time.time()
                for _ in range(3):
                    loss = step(m, imgs, opt)
                torch.npu.synchronize(); dt = (time.time() - t0) / 3
                print(f"{name} ({npar:.0f}M) B={b} N={n} {h}x{w} ({b*n} imgs): {dt:.2f} s/step, "
                      f"peak {torch.npu.max_memory_allocated() / 2**30:.1f} GiB, loss {loss.item():.3f}", flush=True)
            except RuntimeError as e:
                print(f"{name} B={b} N={n} {h}x{w}: FAILED {str(e)[:200]}", flush=True)
        # gradient coverage check on a small input
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            out = m(torch.rand(1, 2, 3, 64, 96, device=dev))
        (out["points"].float().abs().mean()).backward()
        missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        print(f"{name}: params without grad on NPU: {len(missing)} {missing[:3]}", flush=True)
        del m, opt
        torch.npu.empty_cache()
    print("BENCH_DONE")


def ddp():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    dev = torch.device(f"npu:{local}")
    torch.manual_seed(0)
    m = Pi3Loop(**KW, loop_layers=2, loop_k_min=5, loop_k_max=9).to(dev).train()
    dm = DDP(m, device_ids=[local], find_unused_parameters=False)
    opt = torch.optim.AdamW([p for p in dm.parameters() if p.requires_grad], lr=1e-4)
    torch.manual_seed(rank)
    for s in range(3):
        loss = step(dm, torch.rand(1, 8, 3, 336, 512, device=dev), opt)
        ks = [None] * dist.get_world_size()
        dist.all_gather_object(ks, m._loop_calls)
        if rank == 0:
            print(f"ddp step {s}: loss {loss.item():.4f}, K-call counters {ks}", flush=True)
    w = torch.cat([m.decoder[1].mlp.fc1.weight.detach().float().flatten(), m.decoder[2].gate.proj[2].weight.detach().float().flatten()])
    ws = [torch.zeros_like(w) for _ in range(dist.get_world_size())]
    dist.all_gather(ws, w)
    if rank == 0:
        print("params in sync across ranks:", all(torch.equal(ws[0], x) for x in ws))
        print("DDP_NPU_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    {"bench": bench, "ddp": ddp}[sys.argv[1]]()
