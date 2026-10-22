# ScanNet++-only loop-transformer ablation (exp1022_00 .. exp1022_04)

Question: how much decoder weight sharing costs (or gains) when encoder, heads, data and recipe are fixed.
Reference: DejaView / DVLT (NeurIPS 2026, arXiv 2605.30215). Width is 384 everywhere (DINOv3 ViT-S/16
encoder, no encoder->decoder projection), point / camera heads have 2 blocks at width 384 (9.5M).

| run | model config | decoder | training K | inference K | effective depth |
|---|---|---|---|---|---|
| `exp1022_00` baseline | `pi3_dinov3s_w384` | 36 unshared blocks (18 pairs), 63.9M | 1 | 1 | 18 pairs |
| `exp1022_01` | `pi3_dinov3s_loop_w384_L1_k14to18` | loop body 1 pair, 3.55M + 0.17M gates | [14, 18] | 18 | 18 pairs |
| `exp1022_02` | `pi3_dinov3s_loop_w384_L2_k5to9` | loop body 2 pairs, 7.10M + 0.33M gates | [5, 9] | 9 | 18 pairs |
| `exp1022_03` | `pi3_dinov3s_loop_w384_L6_k2to3` | loop body 6 pairs, 21.31M + 1.00M gates | [2, 3] | 3 | 18 pairs |
| `exp1022_04` | `pi3_dinov3s_loop_w384_L1_k1to18` | loop body 1 pair, 3.55M + 0.17M gates | [1, 18] | 18 | 18 pairs |

All runs have the same forward FLOPs at inference (1.96 T for 1 x 8 views at 336x512).

Loop model (`pi3/models/pi3_loop.py`, `pi3/models/layers/loop_block.py`):
- decoder = `[frame_0, global_0, ..., frame_{L-1}, global_{L-1}]`, applied K times; heads read the last frame / global output;
- every block owns a DVLT-style gate MLP (`IntervalGate`: 2 x 64 sinusoidal features of (t_k, t_{k+1}) -> 64 -> 3 x 384,
  last layer zero-initialized, so all gates start at 1): `x += s_attn * attn`, `x += s_mlp * mlp`, `x = s_out * x`;
- training K = k_min + round((k_max - k_min) * Beta(2, 1)), drawn from a call-counter seed so all ranks use the same K;
  inference K = `model.loop_k_infer`, adjustable within the trained range;
- gradient checkpointing per sub-block, non-reentrant (shared parameters are used K times under DDP).

## Launch (ModelArts, 2 nodes x 8 NPUs = 16)

```bash
bash run_sh_s3/run_scannetpp/run_exp1022_00_base_dec384x36.sh
bash run_sh_s3/run_scannetpp/run_exp1022_01_loop384_L1_k14to18.sh
bash run_sh_s3/run_scannetpp/run_exp1022_02_loop384_L2_k5to9.sh
bash run_sh_s3/run_scannetpp/run_exp1022_03_loop384_L6_k2to3.sh
bash run_sh_s3/run_scannetpp/run_exp1022_04_loop384_L1_k1to18.sh
SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_02_loop384_L2_k5to9.sh   # 1 epoch x 50 iters, from the repo root
```
Needs `ckpts/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (ckpts -> /efs-mi-east4-2/checkpoints);
the script exits if it is missing. Everything except NAME / MODEL lives in `common.sh`, so all runs stay identical.

## Recipe (`configs/train/train_scannetpp_dinov3.yaml`, `configs/data/scannetpp_only.yaml`)
- data: `/efs-m3ga/z00533370/reliev3r_data/scannetpp`, scene split 90% train / 10% held-out test (`split_ratio: 0.9`);
- fixed 8 views per sample, 16 samples (128 images) / NPU -> 2048 images / step on 16 NPUs
  (32 samples / NPU ran out of memory in the full Pi3 loss);
  random resolution 100k-255k px (patch 16);
- 30 epochs x 500 iters = 15k steps ~= 31M image samples; ckpt every 2 epochs;
- test once per epoch on a fixed set: 2000 clips (8 views, 512x336) from the held-out scenes, all of them every time
  (125 iters x 16 NPUs, 1 clip per test batch via `test.max_img_per_gpu: 8`, since metrics use batch element 0);
  no shuffling, and each clip's frames are drawn from an RNG keyed on its sample index, so every evaluation and
  every run sees exactly the same clips;
- AdamW, OneCycle (5% warmup, cosine), decoder / heads lr 1e-4, encoder lr 1e-5; loss weights as exp0603.

Measured on 910B3 (1 NPU, bf16, checkpointing on, model fwd/bwd with a trivial loss): 16 samples x 8 views at 432x592
take ~5.1-5.3 s/step and ~16.5 GiB for all five models; the full Pi3 loss adds dense per-pixel tensors on top
(32 samples / NPU hit ~45 GiB inside the loss and OOMed). ~22 h per run without data-loading stalls.

## Fixes included in this branch
- `pi3/models/dinov3/models/vision_transformer.py`: with `USE_CHECKPOINT=True` the encoder used reentrant checkpointing on a
  list input, so **no gradient reached any DINOv3 parameter** (encoder silently frozen). Now `use_reentrant=False`.
- `datasets/scannetpp_dataset.py`: ignore non-image files (Thumbs.db was a "frame"), handle the DSC 09999->00001 counter
  wrap in frame order, map frame name -> pose row by dict. New cache names (`*_cache_v2.pkl`, `sequence_list_v2.pkl`).
- `utils/eval_utils.py`, `trainers/pi3_trainer.py`: camera metrics were all skipped because the installed evo rejects
  `est_name`; evo calls now drop display-only kwargs, and angle / trajectory metrics are computed independently.
- `datasets/__init__.py`: optional `test.max_img_per_gpu` so the test loader no longer inherits the training batch.
