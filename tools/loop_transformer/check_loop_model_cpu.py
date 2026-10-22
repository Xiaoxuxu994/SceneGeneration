"""CPU functional test for the loop-transformer branch (no pretrained weights, small inputs).

  python tools/loop_transformer/check_loop_model_cpu.py single        # shapes, params, gates, K sampling, backward
  torchrun --nproc_per_node 2 tools/loop_transformer/check_loop_model_cpu.py ddp   # DDP (gloo) + shared params + non-reentrant ckpt
"""

import os
import sys

import torch

sys.path.insert(0, os.environ.get("REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))))
import pi3.models.pi3_training as P  # noqa: E402

# Pretrained DINOv3 weights are not needed for a functional test: build the same ViT-S/16 without loading.
_orig_cfg = P.Pi3._build_backbone_cfg


def _cfg_no_download():
    cfg = _orig_cfg()
    b = cfg["dinov3"]["small"]["builder"]
    cfg["dinov3"]["small"]["builder"] = lambda pretrained=True, model_dir=None: b(pretrained=False)
    return cfg


P.Pi3._build_backbone_cfg = staticmethod(_cfg_no_download)
from pi3.models.pi3_loop import Pi3Loop  # noqa: E402

KW = dict(pos_type="rope100", model_size="dinov3_small_1B", load_vggt=False, freeze_encoder=False,
          use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=0, ckpt=None)


def nparams(m, prefix=None):
    return sum(p.numel() for n, p in m.named_parameters() if prefix is None or n.startswith(prefix)) / 1e6


def single():
    torch.manual_seed(0)
    A = P.Pi3(**KW)
    B = Pi3Loop(**KW, loop_k_min=12, loop_k_max=18, loop_k_infer=18, loop_k_beta=[2.0, 1.0], loop_seed=0)
    for name, m in [("A", A), ("B", B)]:
        print(f"{name}: total {nparams(m):.1f}M | encoder {nparams(m, 'encoder'):.1f}M | decoder {nparams(m, 'decoder'):.1f}M | "
              f"heads {nparams(m, 'point_') + nparams(m, 'camera_'):.1f}M | time_embed {nparams(m, 'time_embed'):.2f}M")

    # gates are exactly 1 at init
    g = B.time_embed(torch.tensor(0.0), torch.tensor(1 / 18))
    assert g.shape == (6, 1024) and torch.equal(g, torch.ones_like(g)), "gates must be 1 at init"

    # K sampling: deterministic across instances (= ranks), within range, skewed to k_max
    # K depends only on (loop_seed, call counter) -> replaying the counter must give the same sequence
    B.train(); ks = [B.sample_k() for _ in range(2000)]
    B._loop_calls = 0; ks2 = [B.sample_k() for _ in range(2000)]
    assert ks == ks2, "K sequence must be identical on every rank"
    B._loop_calls = 0
    hist = {k: ks.count(k) for k in range(12, 19)}
    print("K histogram (train):", hist, "mean", sum(ks) / len(ks))
    assert min(ks) >= 12 and max(ks) <= 18
    B.eval(); assert B.sample_k() == 18

    # forward / backward, both models, small input (B=1, N=3, 64x96 -> 4x6 patches)
    imgs = torch.rand(1, 3, 3, 64, 96)
    for name, m in [("A", A), ("B", B)]:
        m.train()
        out = m(imgs)
        assert set(out) == {"points", "local_points", "conf", "camera_poses", "global_points"}
        assert out["points"].shape == (1, 3, 64, 96, 3) and out["camera_poses"].shape == (1, 3, 4, 4)
        loss = out["points"].abs().mean() + out["camera_poses"].abs().mean()
        loss.backward()
        gnorm = {n: p.grad.norm().item() for n, p in m.named_parameters() if p.grad is not None and n.startswith(("decoder.0.attn.qkv.weight", "decoder.1.mlp.fc2.weight", "time_embed.mlp.2.weight", "encoder.blocks.0.attn.qkv.weight"))}
        none = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        print(f"{name}: forward/backward ok, params without grad: {len(none)} {none[:3]}; grad norms {gnorm}")
        m.zero_grad()

    # inference with different K (test-time compute knob)
    B.eval()
    with torch.no_grad():
        for k in (12, 18):
            B.loop_k_infer = k
            o = B(imgs)
            print(f"B eval K={k}: depth mean {o['local_points'][..., 2].mean():.4f}")
    B.loop_k_infer = 18

    # state_dict round trip (checkpoint save / resume)
    sd = B.state_dict()
    B3 = Pi3Loop(**KW)
    print("B state_dict reload:", B3.load_state_dict(sd, strict=True))
    print("SINGLE_OK")


def ddp():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)
    m = Pi3Loop(**KW, loop_k_min=12, loop_k_max=18)
    m.train()
    dm = DDP(m, find_unused_parameters=False)
    opt = torch.optim.AdamW([p for p in dm.parameters() if p.requires_grad], lr=1e-4)
    torch.manual_seed(rank)  # different data per rank
    for step in range(2):
        imgs = torch.rand(1, 2, 3, 64, 64)
        out = dm(imgs)
        loss = out["points"].abs().mean() + out["camera_poses"].abs().mean()
        loss.backward()
        opt.step(); opt.zero_grad()
        ks = [None, None]
        dist.all_gather_object(ks, m._loop_calls)
        if rank == 0:
            print(f"step {step}: loss {loss.item():.4f}, loop call counters per rank {ks}")
    # parameters stay identical across ranks after DDP steps
    w = m.decoder[0].attn.qkv.weight.detach().clone()
    ws = [torch.zeros_like(w) for _ in range(dist.get_world_size())]
    dist.all_gather(ws, w)
    if rank == 0:
        print("params in sync across ranks:", all(torch.equal(ws[0], x) for x in ws))
        print("DDP_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    {"single": single, "ddp": ddp}[sys.argv[1]]()
