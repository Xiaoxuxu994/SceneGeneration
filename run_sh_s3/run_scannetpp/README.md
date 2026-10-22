# ScanNet++-only loop-transformer study

Goal: a looped (weight-shared) decoder reaches the accuracy of the unshared Pi3-1B decoder when both use the
same DINOv3 ViT-S/16 encoder, heads, data and recipe. Reference: DejaView / DVLT (NeurIPS 2026, arXiv 2605.30215).

| | A `pi3_dinov3s_1b` | B `pi3_dinov3s_loop` |
|---|---|---|
| encoder | DINOv3 ViT-S/16 (21.6M) | same |
| projection | Linear 384->1024 + LN | same |
| decoder | 36 unshared BlockRope (1024, 16 heads), frame/global alternating — 453.5M | 1 frame + 1 global GatedBlockRope looped K times — 25.2M + 3.4M time gates |
| heads | Pi3 point / camera decoders (5 blocks) — 134.6M | same |
| total | 610.2M | 185.2M |
| decoder FLOPs | 36 blocks | K = 18 -> identical to A |

B details (`pi3/models/pi3_loop.py`, `pi3/models/layers/loop_block.py`):
- gates `s_attn, s_mlp, s_out = 1 + MLP(gamma(t_k, t_{k+1}))`, t_k = k / K, MLP zero-initialized (gates = 1 at init);
- training K = 12 + round(6 * Beta(2, 1)) (mean ~16), drawn from a call-counter seed so all ranks use the same K;
  inference K = `model.loop_k_infer` (18), adjustable within the trained range;
- gradient checkpointing per sub-block, non-reentrant (shared parameters are used K times under DDP);
- block parameter names equal BlockRope, so a Pi3 decoder block can be loaded into the shared blocks later.

## Launch (ModelArts, 4 nodes x 8 NPUs = 32)

```bash
bash run_sh_s3/run_scannetpp/run_A_dinov3s_pi3dec1b.sh
bash run_sh_s3/run_scannetpp/run_B_dinov3s_loop.sh
SMOKE=1 bash run_sh_s3/run_scannetpp/run_B_dinov3s_loop.sh   # 1 epoch x 50 iters sanity run
```
Needs `ckpts/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (ckpts -> /efs-mi-east4-2/checkpoints);
the script exits if it is missing. Everything except NAME / MODEL lives in `common.sh`, so A and B stay identical.

## Recipe (`configs/train/train_scannetpp_dinov3.yaml`, `configs/data/scannetpp_only.yaml`)
- data: `/efs-m3ga/z00533370/reliev3r_data/scannetpp`, scene split 90% train / 10% held-out test (`split_ratio: 0.9`);
- 2-16 views per sample, random resolution 100k-255k px (patch 16), up to 32 images / NPU (~1024 / step);
- 30 epochs x 1000 iters = 30k steps ~= 31M image samples; held-out test every epoch (50 iters); ckpt every 2 epochs;
- AdamW, OneCycle (5% warmup, cosine), decoder / heads lr 1e-4, encoder lr 1e-5; loss weights as exp0603.

## Measured on 910B3 (1 NPU, bf16, random init, K = 18)
| shape (images / NPU) | A s/step | A peak | B s/step | B peak |
|---|---|---|---|---|
| 2 x 16 views, 432x592 (32) | 4.40 | 15.7 GiB | 4.52 | 15.3 GiB |
| 4 x 8 views, 432x592 (32) | 3.44 | 15.7 GiB | 3.55 | 15.3 GiB |
| 1 x 16 views, 336x512 (16) | 1.28 | 11.5 GiB | 1.36 | 9.9 GiB |

Without decoder checkpointing 32 images / NPU OOM (> 61 GiB) and 16 images are only ~20% faster per image, so
checkpointing stays on. Expected wall time ~30 h per run (plus data loading). HCCL DDP with shared parameters
verified on 2 NPUs (`tools/loop_transformer/check_loop_model_npu.py ddp`).

## Fixes included in this branch
- `pi3/models/dinov3/models/vision_transformer.py`: with `USE_CHECKPOINT=True` the encoder used reentrant checkpointing on a
  list input, so **no gradient reached any DINOv3 parameter** (encoder silently frozen). Now `use_reentrant=False`.
- `datasets/scannetpp_dataset.py`: ignore non-image files (Thumbs.db was a "frame"), handle the DSC 09999->00001 counter
  wrap in frame order, map frame name -> pose row by dict (the `searchsorted(..., sorter=...)` lookup only worked because
  metadata happened to be sorted). New cache names (`*_cache_v2.pkl`, `sequence_list_v2.pkl`) force a rebuild.
