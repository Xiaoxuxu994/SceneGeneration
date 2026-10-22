"""Pi3 with a looped (weight-shared) decoder, after DejaView / DVLT (NeurIPS 2026).

Same encoder, projection, register tokens, point / camera heads and outputs as `Pi3`; only the
decoder changes: instead of `dec_depth` independent blocks (alternating frame / global attention),
a loop body of L (frame, global) block pairs is applied K times with shared weights. Every block owns a
DVLT-style gate MLP conditioned on the time interval (t_k, t_{k+1}) = (k/K, (k+1)/K), zero-initialized.

  training : K = k_min + round((k_max - k_min) * Beta(a, b)), identical on every rank
  inference: K = loop_k_infer (default k_max; can be changed at test time for compute scaling)

L * K = dec_depth / 2 (18 for a 36-block decoder) gives exactly the decoder FLOPs of the unshared model.
decoder = [frame_0, global_0, frame_1, global_1, ...]; the blocks keep BlockRope parameter names.
"""

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import load_sharded_checkpoint

from .dinov2.layers import Mlp
from .layers.attention import FlashAttentionRope
from .layers.loop_block import GatedBlockRope
from .pi3_training import Pi3


class Pi3Loop(Pi3):
    def __init__(
        self,
        pos_type='rope100',
        decoder_size='large',
        model_size=None,
        load_vggt=False,
        freeze_encoder=False,
        use_global_points=False,
        train_conf=False,
        num_dec_blk_not_to_checkpoint=0,
        ckpt=None,
        loop_layers=1,
        loop_k_min=12,
        loop_k_max=18,
        loop_k_infer=None,
        loop_k_beta=(2.0, 1.0),
        loop_gates=True,
        loop_gate_hidden=64,
        loop_grad_checkpoint=True,
        loop_seed=0,
    ):
        # Build everything except the checkpoint load with the parent; the decoder is replaced below.
        super().__init__(pos_type=pos_type, decoder_size=decoder_size, model_size=model_size, load_vggt=False,
                         freeze_encoder=freeze_encoder, use_global_points=use_global_points, train_conf=False,
                         num_dec_blk_not_to_checkpoint=num_dec_blk_not_to_checkpoint, ckpt=None)
        assert not load_vggt, "Pi3Loop: VGGT decoder init is not supported (use a Pi3 checkpoint instead)"
        assert not train_conf, "Pi3Loop: confidence-head training is not supported yet"

        ref = self.decoder[0]
        dim = self.dec_embed_dim
        num_heads = ref.attn.num_heads
        mlp_ratio = ref.mlp.fc1.out_features / dim
        del self.decoder  # drop the dec_depth unshared blocks

        def make_block():
            return GatedBlockRope(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, proj_bias=True, ffn_bias=True,
                drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, ffn_layer=Mlp,
                init_values=0.01, qk_norm=True, attn_class=FlashAttentionRope, rope=self.rope,
                gate_hidden_dim=int(loop_gate_hidden), use_gate=loop_gates)

        # loop body: decoder[2l] = frame-attention block, decoder[2l + 1] = global-attention block, l < L
        assert loop_layers >= 1
        self.loop_layers = int(loop_layers)
        self.decoder = nn.ModuleList([make_block() for _ in range(2 * self.loop_layers)])
        self.loop_gates = loop_gates

        assert 1 <= loop_k_min <= loop_k_max
        self.loop_k_min, self.loop_k_max = int(loop_k_min), int(loop_k_max)
        self.loop_k_infer = int(loop_k_infer) if loop_k_infer is not None else self.loop_k_max
        self.loop_k_beta = tuple(float(x) for x in loop_k_beta)
        self.loop_grad_checkpoint = loop_grad_checkpoint
        self.loop_seed = int(loop_seed)
        self._loop_calls = 0  # advances identically on every rank -> same K across DDP ranks

        if ckpt is not None:
            ckpt = Path(ckpt)
            if ckpt.is_dir():
                res = load_sharded_checkpoint(self, str(ckpt), strict=False)
            else:
                state = torch.load(str(ckpt), weights_only=False, map_location="cpu")
                res = self.load_state_dict(state, strict=False)
                del state
            print(f"[Pi3Loop] Load checkpoints from {ckpt}: {res}")

        n_gate = sum(p.numel() for n, p in self.decoder.named_parameters() if '.gate.' in n)
        n_dec = sum(p.numel() for p in self.decoder.parameters()) - n_gate
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[Pi3Loop] loop body {self.loop_layers} pair(s): blocks {n_dec / 1e6:.2f}M + gates {n_gate / 1e6:.2f}M params, "
              f"K train in [{self.loop_k_min}, {self.loop_k_max}] Beta{self.loop_k_beta}, K infer {self.loop_k_infer}, "
              f"total {n_all / 1e6:.2f}M params")

    def sample_k(self):
        if not self.training:
            return self.loop_k_infer
        if self.loop_k_min == self.loop_k_max:
            return self.loop_k_max
        # Seeded by a call counter (not the global RNG) so every DDP rank draws the same K.
        g = torch.Generator().manual_seed(self.loop_seed * 1_000_003 + self._loop_calls)
        self._loop_calls += 1
        # Beta(a, b) = Ga / (Ga + Gb); torch.distributions.Beta cannot take a generator.
        gam = torch._standard_gamma(torch.tensor(self.loop_k_beta, dtype=torch.float64), generator=g)
        beta = float(gam[0] / gam.sum())
        return self.loop_k_min + int(round((self.loop_k_max - self.loop_k_min) * beta))

    def _run(self, blk, x, pos, xpos_max, gates):
        s_attn, s_mlp, s_out = gates
        if self.loop_grad_checkpoint and self.training:
            # non-reentrant: the shared parameters are used K times inside one DDP forward
            return checkpoint(blk.forward_gated, x, s_attn, s_mlp, s_out, pos, xpos_max, use_reentrant=False)
        return blk.forward_gated(x, s_attn, s_mlp, s_out, xpos=pos, xpos_max=xpos_max)

    def decode(self, hidden, N, H, W, xpos_max=None):
        BN, _, C = hidden.shape
        B = BN // N
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos = torch.cat([self.pos_special.repeat(B * N, 1, 1), pos], dim=1)
        pos_frame = pos.reshape(B * N, hw, -1)
        pos_global = pos.reshape(B, N * hw, -1)

        K = self.sample_k()
        for k in range(K):
            t0, t1 = k / K, (k + 1) / K
            for l in range(self.loop_layers):
                frame_blk, global_blk = self.decoder[2 * l], self.decoder[2 * l + 1]
                hidden = self._run(frame_blk, hidden.reshape(B * N, hw, C), pos_frame, xpos_max,
                                   frame_blk.gates(t0, t1, hidden))
                frame_out = hidden.reshape(B * N, hw, C)
                hidden = self._run(global_blk, hidden.reshape(B, N * hw, C), pos_global, xpos_max,
                                   global_blk.gates(t0, t1, hidden))
        global_out = hidden.reshape(B * N, hw, C)
        # same head interface as Pi3: [last frame-block output, last global-block output]
        return torch.cat([frame_out, global_out], dim=-1), pos_frame
