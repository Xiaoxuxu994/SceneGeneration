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

KW = dict(pos_type="rope100", model_size=os.environ.get("MODEL_SIZE", "dinov3_small_w384"), load_vggt=False, freeze_encoder=False,
          use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=0, ckpt=None)


def nparams(m, prefix=None):
    return sum(p.numel() for n, p in m.named_parameters() if prefix is None or n.startswith(prefix)) / 1e6


# the four loop experiments: (L, k_min, k_max), inference K = k_max -> L * k_max = 18 pairs
LOOPS = [(1, 14, 18), (2, 5, 9), (6, 2, 3), (1, 1, 18)]


def single():
    torch.manual_seed(0)
    A = P.Pi3(**KW)
    print(f"base: total {nparams(A):.2f}M | encoder {nparams(A, 'encoder'):.2f}M | decoder {nparams(A, 'decoder'):.2f}M | "
          f"heads {nparams(A, 'point_') + nparams(A, 'camera_'):.2f}M")
    imgs = torch.rand(1, 3, 3, 64, 96)  # B=1, N=3, 64x96 -> 4x6 patches
    models = [("base", A)]
    for (L, kmin, kmax) in LOOPS:
        name = f"L{L}_k{kmin}to{kmax}"
        B = Pi3Loop(**KW, loop_layers=L, loop_k_min=kmin, loop_k_max=kmax, loop_k_infer=kmax, loop_seed=0)
        n_gate = sum(p.numel() for n, p in B.named_parameters() if ".gate." in n) / 1e6
        print(f"{name}: total {nparams(B):.2f}M | decoder blocks {nparams(B, 'decoder') - n_gate:.2f}M | gates {n_gate:.3f}M | "
              f"{len(B.decoder)} blocks")
        assert len(B.decoder) == 2 * L

        # gates are exactly 1 at init, one gate MLP per block
        for blk in B.decoder:
            g = blk.gates(0.0, 1 / kmax, torch.zeros(1, B.dec_embed_dim))
            assert g.shape == (3, B.dec_embed_dim) and torch.equal(g, torch.ones_like(g)), "gates must be 1 at init"

        # K sampling: identical on every rank (depends only on seed + call counter), within range, skewed to k_max
        B.train(); ks = [B.sample_k() for _ in range(2000)]
        B._loop_calls = 0; ks2 = [B.sample_k() for _ in range(2000)]
        assert ks == ks2, "K sequence must be identical on every rank"
        B._loop_calls = 0
        assert min(ks) >= kmin and max(ks) <= kmax
        print(f"  K histogram (train): { {k: ks.count(k) for k in range(kmin, kmax + 1)} } mean {sum(ks) / len(ks):.2f}, "
              f"mean effective pairs {L * sum(ks) / len(ks):.1f}")
        B.eval(); assert B.sample_k() == kmax
        models.append((name, B))

    for name, m in models:
        m.train()
        out = m(imgs)
        assert set(out) == {"points", "local_points", "conf", "camera_poses", "global_points"}
        assert out["points"].shape == (1, 3, 64, 96, 3) and out["camera_poses"].shape == (1, 3, 4, 4)
        loss = out["points"].abs().mean() + out["camera_poses"].abs().mean()
        loss.backward()
        none = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        # the zero-initialized last gate layer must get a gradient, otherwise the gates never move
        gate_g = [p.grad.norm().item() for n, p in m.named_parameters() if n.endswith("gate.proj.2.weight")]
        print(f"{name}: forward/backward ok, params without grad: {len(none)} {none[:3]}; "
              f"gate last-layer grad norms {[round(g, 6) for g in gate_g[:4]]}")
        assert not none
        assert all(g > 0 for g in gate_g)
        m.zero_grad()

    # inference with different K (test-time compute knob) and state_dict round trip
    B = models[1][1]
    B.eval()
    with torch.no_grad():
        for k in (14, 18):
            B.loop_k_infer = k
            o = B(imgs)
            print(f"L1 eval K={k}: depth mean {o['local_points'][..., 2].mean():.4f}")
    B.loop_k_infer = 18
    sd = models[2][1].state_dict()
    B3 = Pi3Loop(**KW, loop_layers=2, loop_k_min=5, loop_k_max=9)
    print("L2 state_dict reload:", B3.load_state_dict(sd, strict=True))
    print("SINGLE_OK")


def ddp():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)
    m = Pi3Loop(**KW, loop_layers=2, loop_k_min=5, loop_k_max=9)
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
    w = torch.cat([m.decoder[0].attn.qkv.weight.detach().flatten(), m.decoder[3].gate.proj[2].weight.detach().flatten()])
    ws = [torch.zeros_like(w) for _ in range(dist.get_world_size())]
    dist.all_gather(ws, w)
    if rank == 0:
        print("params in sync across ranks:", all(torch.equal(ws[0], x) for x in ws))
        print("DDP_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    {"single": single, "ddp": ddp}[sys.argv[1]]()
