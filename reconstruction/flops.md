# Reliev3R Inference FLOPs Analysis (Corrected)

**Configuration**: `configs/reliev3r_quant.json` (`depth_aligner.pi3`, `decoder_size="large"` = 0.95B)
**Scenario**: Single batch, B=1, V=8 views
**Date**: 2026-04-15

---

## Architecture Overview

| Stage | Module | Processing | Parameters |
|-------|--------|------------|------------|
| **Encoder** | DINOv2 ViT-L/14 (w/ registers) | Per-image, V times | embed_dim=1024, depth=24, heads=16, mlp_ratio=4, patch_size=14, 4 reg tokens |
| **Projection** | Identity (enc_dim == dec_dim) | — | 0 FLOPs |
| **Decoder (backbone)** | 36 BlockRope blocks | Even: per-view (B\*V, n_pv); Odd: cross-view (B, V\*n_pv) | dim=1024, heads=16, mlp_ratio=4, 5 reg/view, qk_norm, LayerScale |
| **Depth Head** | TransformerDecoder(5 blk) + LinearPts3d | Per-view (B\*V, n_pv) | in=2048, internal=1024, out=1024, no qk_norm, no LayerScale |
| **Conf Head** | TransformerDecoder(5 blk) + LinearPts3d | Per-view (B\*V, n_pv) | in=2048, internal=1024, out=1024, no qk_norm, no LayerScale |
| **Camera Head** | TransformerDecoder(5 blk) + CameraHead | Per-view (B\*V, n_pv) | in=2048, internal=1024, out=512, no qk_norm, no LayerScale |

---

## Token Counts

| Resolution | Patch Grid | Patches | Encoder Tokens* | Per-View Tokens** | Cross-View Tokens*** |
|-----------|-----------|---------|-----------------|-------------------|---------------------|
| 518x518   | 37x37     | 1,369   | 1,374           | 1,374             | 10,992              |
| 1080x1920 | 78x138    | 10,764  | 10,769          | 10,769            | 86,152              |

\* Patches + 1 CLS + 4 register tokens (encoder)
\** Patches + 5 register tokens (decoder per-view, and head decoder)
\*** V x per-view tokens = 8 x per-view (decoder cross-view odd blocks only)

---

## FLOPs Convention

- 1 MAC (multiply-accumulate) = 2 FLOPs
- `Linear(in, out)` = 2 x seq_len x in x out
- **Dense attention**: 2 x N^2 x d (QK^T) + 2 x N^2 x d (attn@V) = **4 x N^2 x d**
- **Sparse attention** (FlashAttentionRope.sparsity=s>0): Q stays at N, K/V subsampled to N//s
  → 2 x N x (N/s) x d (QK^T) + 2 x N x (N/s) x d (attn@V) = **4 x N x (N/s) x d**
- All values in **GFLOPs** (10^9 floating-point operations)
- `FlashAttentionRope.sparsity`: class-level attribute (0 = dense, N = subsample factor)
- **Sparsity scope**: In the benchmark, `patch_sparse_attention()` only applies non-zero sparsity to **odd-numbered (cross-view) backbone decoder blocks**. Even-numbered (per-view) backbone blocks and all prediction-head blocks are always dense.

---

## Per-Block FLOPs

### Backbone Block (has qk_norm + LayerScale)

| Operation | Formula | 518x518 n=1374 (MF) | 518x518 n=10992 (MF) |
|-----------|---------|--------------------|--------------------|
| LayerNorm x2 | 4 x n x d x 2 | 11.26 | 90.04 |
| QKV Projection | 2 x n x d x 3d | 8,644.46 | 69,155.68 |
| Attention (dense) | 4 x n^2 x d | 7,732.74 | 494,895.37 |
| QK Norm | 4 x n x d | 5.63 | 45.02 |
| RoPE | 4 x n x d | 5.63 | 45.02 |
| Output Projection | 2 x n x d x d | 2,881.49 | 23,051.89 |
| LayerScale x2 | 2 x n x d x 2 | 5.63 | 45.02 |
| MLP (fc1 + fc2) | 2 x n x d x 4d x 2 | 23,051.90 | 184,415.16 |
| **Block Total** | | **42,338.74** | **771,743.20** |

### Head Decoder Block (no qk_norm, no LayerScale)

| Operation | Formula | 518x518 n=1374 (MF) |
|-----------|---------|--------------------|
| LayerNorm x2 | 4 x n x d x 2 | 11.26 |
| QKV Projection | 2 x n x d x 3d | 8,644.46 |
| Attention (dense) | 4 x n^2 x d | 7,732.74 |
| RoPE | 4 x n x d | 5.63 |
| Output Projection | 2 x n x d x d | 2,881.49 |
| MLP (fc1 + fc2) | 2 x n x d x 4d x 2 | 23,051.90 |
| **Block Total** | | **42,327.48** |

> The per-block difference is ~11 MFLOPs (qk_norm + LayerScale ≈ 0.03% overhead).

---

## 518x518 Resolution

### Stage 1: DINOv2-L/14 Encoder (x8 images, per-image)

| Operation | FLOPs/Block (GF) | x24 Blocks (GF) |
|-----------|-----------------|-----------------|
| PatchEmbed Conv | 1.649 (one-time) | — |
| LayerNorm x2 | 0.006 x 2 | 0.271 |
| QKV Projection | 8.644 | 207.467 |
| Attention (1374^2) | 7.733 | 185.586 |
| Output Projection | 2.881 | 69.156 |
| LayerScale x2 | 0.003 x 2 | 0.135 |
| MLP fc1 | 11.526 | 276.623 |
| MLP fc2 | 11.526 | 276.623 |
| Final LayerNorm | 0.006 (one-time) | — |
| **Per-image Total** | | **1,017.51** |
| **Encoder Total (x8 views)** | | **8,140.11** |

### Stage 3: 36-Block Pi3 Backbone Decoder

#### Even Blocks (Per-View Self-Attention, n=1,374)

| Operation | FLOPs/Block (GF) | x18 Blocks (GF) |
|-----------|-----------------|-----------------|
| LayerNorm x2 | 0.006 x 2 | 0.203 |
| QKV Projection | 8.644 | 155.600 |
| Attention (1374^2) | 7.733 | 139.192 |
| QK Norm | 0.006 | 0.101 |
| RoPE | 0.006 | 0.101 |
| Output Projection | 2.881 | 51.867 |
| LayerScale x2 | 0.003 x 2 | 0.101 |
| MLP fc1+fc2 | 23.052 | 414.934 |
| **Even Block Total** | **42.339** | **762.10** |

#### Odd Blocks (Cross-View Attention, n=10,992 = 8x1,374)

| Operation | FLOPs/Block sparsity=0 (GF) | x18 Blocks sparsity=0 (GF) |
|-----------|---------------------|---------------------|
| LayerNorm x2 | 0.045 x 2 | 1.621 |
| QKV Projection | 69.156 | 1,244.802 |
| Attention (10992^2) | **494.895** | **8,908.117** |
| QK Norm | 0.045 | 0.811 |
| RoPE | 0.045 | 0.811 |
| Output Projection | 23.052 | 414.934 |
| LayerScale x2 | 0.023 x 2 | 0.811 |
| MLP fc1+fc2 | 184.415 | 3,319.475 |
| **Odd Block Total** | **771.743** | **13,891.38** |

| **Backbone Decoder Total (sparsity=0)** | | **14,653.47** |

### Stage 4: Prediction Heads (per-view, n=1,374, batch=B\*V=8)

Each head receives `(B*V, n_pv, 2*d)` from the backbone decoder, projects to `(B*V, n_pv, 1024)`,
then runs 5 per-view blocks.

| Head | Projection (GF) | 5 Blocks (GF) | Linear Out (GF) | Total (GF) |
|------|----------------|--------------|----------------|-----------|
| Depth Decoder | 5.76 | 211.64 | 2.88 | 220.28 |
| Conf Decoder | 5.76 | 211.64 | 2.88 | 220.28 |
| Camera Decoder | 5.76 | 211.64 | 1.44 | 218.84 |
| **Heads Total** | | | | **659.40** |

### Stage 5: Output Projections

| Operation | FLOPs (GF) |
|-----------|-----------|
| Depth Head Linear (1024->196, x8 views x 1369 patches) | 4.40 |
| Conf Head Linear (1024->196) | 4.40 |
| Camera ResConvBlock x2 + MLPs + fc | 34.47 |
| **Stage 5 Total** | **43.26** |

### Grand Total -- 518x518 (sparsity=0)

| Stage | FLOPs (GF) | % of Total |
|-------|-----------|-----------|
| Encoder (DINOv2-L/14) | 8,140.11 | 34.6% |
| Projection | 0.00 | 0.0% |
| **Backbone Decoder (36 blocks)** | **14,653.47** | **62.4%** |
| -- Even (per-view) | 762.10 | 3.2% |
| -- Odd (cross-view) | 13,891.38 | 59.1% |
| Prediction Heads (per-view) | 659.40 | 2.8% |
| Output Projections | 43.26 | 0.2% |
| **TOTAL** | **23,496.25** | **100%** |

---

## 1080x1920 Resolution

### Stage 1: DINOv2-L/14 Encoder (x8 images, per-image)

| Operation | FLOPs/Block (GF) | x24 Blocks (GF) |
|-----------|-----------------|-----------------|
| PatchEmbed Conv | 12.962 (one-time) | — |
| LayerNorm x2 | 0.044 x 2 | 2.118 |
| QKV Projection | 67.753 | 1,626.065 |
| Attention (10769^2) | 475.019 | 11,400.449 |
| Output Projection | 22.584 | 542.022 |
| LayerScale x2 | 0.022 x 2 | 1.058 |
| MLP fc1 | 90.337 | 2,168.086 |
| MLP fc2 | 90.337 | 2,168.086 |
| Final LayerNorm | 0.044 (one-time) | — |
| **Per-image Total** | | **17,920.84** |
| **Encoder Total (x8 views)** | | **143,367.11** |

### Stage 3: 36-Block Pi3 Backbone Decoder

#### Even Blocks (Per-View, n=10,769)

| Operation | FLOPs/Block (GF) | x18 Blocks (GF) |
|-----------|-----------------|-----------------|
| LayerNorm x2 | 0.044 x 2 | 1.588 |
| QKV Projection | 67.753 | 1,219.548 |
| Attention (10769^2) | 475.019 | 8,550.340 |
| QK Norm | 0.044 | 0.794 |
| RoPE | 0.044 | 0.794 |
| Output Projection | 22.584 | 406.516 |
| LayerScale x2 | 0.022 x 2 | 0.794 |
| MLP fc1+fc2 | 180.674 | 3,252.131 |
| **Even Block Total** | **746.250** | **13,432.50** |

#### Odd Blocks (Cross-View, n=86,152 = 8x10,769)

| Operation | FLOPs/Block sparsity=0 (GF) | x18 Blocks sparsity=0 (GF) |
|-----------|---------------------|---------------------|
| LayerNorm x2 | 0.353 x 2 | 12.704 |
| QKV Projection | 542.022 | 9,756.388 |
| Attention (86152^2) | **30,401.197** | **547,221.537** |
| QK Norm | 0.353 | 6.352 |
| RoPE | 0.353 | 6.352 |
| Output Projection | 180.674 | 3,252.127 |
| LayerScale x2 | 0.176 x 2 | 6.352 |
| MLP fc1+fc2 | 1,445.391 | 26,017.034 |
| **Odd Block Total** | **32,571.047** | **586,278.84** |

| **Backbone Decoder Total (sparsity=0)** | | **599,711.34** |

### Stage 4: Prediction Heads (per-view, n=10,769, batch=B\*V=8)

| Head | Projection (GF) | 5 Blocks (GF) | Linear Out (GF) | Total (GF) |
|------|----------------|--------------|----------------|-----------|
| Depth Decoder | 45.17 | 3,730.81 | 22.58 | 3,798.56 |
| Conf Decoder | 45.17 | 3,730.81 | 22.58 | 3,798.56 |
| Camera Decoder | 45.17 | 3,730.81 | 11.29 | 3,787.27 |
| **Heads Total** | | | | **11,384.39** |

### Stage 5: Output Projections

| Operation | FLOPs (GF) |
|-----------|-----------|
| Depth Head Linear (1024->196, x8 views x 10764 patches) | 34.57 |
| Conf Head Linear (1024->196) | 34.57 |
| Camera ResConvBlock x2 + MLPs + fc | 270.90 |
| **Stage 5 Total** | **340.03** |

### Grand Total -- 1080x1920 (sparsity=0)

| Stage | FLOPs (GF) | % of Total |
|-------|-----------|-----------|
| Encoder (DINOv2-L/14) | 143,367.11 | 19.0% |
| Projection | 0.00 | 0.0% |
| **Backbone Decoder (36 blocks)** | **599,711.34** | **79.5%** |
| -- Even (per-view) | 13,432.50 | 1.8% |
| -- Odd (cross-view) | 586,278.84 | 77.7% |
| Prediction Heads (per-view) | 11,384.39 | 1.5% |
| Output Projections | 340.03 | 0.0% |
| **TOTAL** | **754,802.88** | **100%** |

---

## Cross-View Attention Sparsity Comparison

`FlashAttentionRope.sparsity` is a class-level attribute. When set to `s > 0`, it subsamples **K and V** to `N // s` tokens (Q stays at N, output shape stays `(B, N, C)`).

> **Benchmark behavior**: In the actual inference pipeline (`patch_sparse_attention()` in `benchmark/lib/quant.py`), sparsity is applied **only to odd-numbered (cross-view) decoder blocks** in the Pi3 backbone. Per-view attention — even-numbered backbone blocks and all prediction-head blocks — is **always dense (sparsity=0)**.

### Sparsity Effect on Attention FLOPs

Corrected formula: `4 * N * (N/s) * d` (Q full, KV subsampled)
Previous (incorrect): `4 * (N/s)^2 * d` (assumed Q also subsampled)

| sparsity | 518x518 Cross-View (n=10992) | | 1080x1920 Cross-View (n=86152) | |
|:---:|---:|:---:|---:|:---:|
| | Attn/Block (GF) | Ratio | Attn/Block (GF) | Ratio |
| 0 (original) | 494.90 | 1.000 | 30,401.20 | 1.000 |
| 2 | 247.45 | 0.500 | 15,200.60 | 0.500 |
| 4 | 123.72 | 0.250 | 7,600.30 | 0.250 |
| 8 | 61.86 | 0.125 | 3,800.15 | 0.125 |

### Sparsity Effect on Odd-Block Total (x18 blocks) -- Cross-View Only

| sparsity | 518x518 | | | 1080x1920 | | |
|:---:|---:|---:|:---:|---:|---:|:---:|
| | Odd Total (GF) | Grand (GF) | Savings | Odd Total (GF) | Grand (GF) | Savings |
| 0 | 13,891.38 | 23,496.25 | -- | 586,278.84 | 754,802.88 | -- |
| 2 | 9,437.32 | 19,042.19 | **19.0%** | 312,668.08 | 481,192.12 | **36.2%** |
| 4 | 7,210.29 | 16,815.16 | **28.4%** | 175,862.69 | 344,386.73 | **54.4%** |
| 8 | 6,096.78 | 15,701.65 | **33.2%** | 107,460.00 | 275,984.04 | **63.4%** |

> **Scope note**: In the benchmark code (`patch_sparse_attention()` in `benchmark/lib/quant.py`), sparsity is only applied to odd-numbered cross-view decoder blocks. Per-view attention (even backbone blocks + all head blocks) is always dense regardless of the sparsity parameter. The above table reflects this selective patching behavior.

### Per-Block Attention Cost Comparison

| | sparsity=0 | sparsity=2 | sparsity=4 | sparsity=8 |
|---|---:|---:|---:|---:|
| **518x518** | | | | |
| Cross-view (n=10992) | 494.90 | 247.45 | 123.72 | 61.86 |
| Per-view backbone (n=1374) | 7.73 | 7.73 | 7.73 | 7.73 |
| Per-view heads (n=1374) | 7.73 | 7.73 | 7.73 | 7.73 |
| **1080x1920** | | | | |
| Cross-view (n=86152) | 30,401.20 | 15,200.60 | 7,600.30 | 3,800.15 |
| Per-view backbone (n=10769) | 475.02 | 475.02 | 475.02 | 475.02 |
| Per-view heads (n=10769) | 475.02 | 475.02 | 475.02 | 475.02 |

*(All values in GFLOPs. Per-view rows are always dense — sparsity setting does not affect them.)*

---

## Summary

### Resolution Scaling

| Resolution | Total FLOPs (GF) | Encoder | Backbone Even | Backbone Odd (cross-view) | Heads | TFLOPs |
|-----------|-----------------|---------|--------------|--------------------------|-------|--------|
| 518x518 | 23,496 | 8,140 (34.6%) | 762 (3.2%) | 13,891 (59.1%) | 659 (2.8%) | 23.5 |
| 1080x1920 | 754,803 | 143,367 (19.0%) | 13,433 (1.8%) | 586,279 (77.7%) | 11,384 (1.5%) | 754.8 |

The 1080x1920 resolution is **32.1x** more expensive than 518x518 (dominated by cross-view attention scaling quadratically: (86152/10992)^2 = 61.3x for odd-block attention).

### Key Bottleneck: Cross-View Attention in Pi3 Backbone

Cross-view attention in the odd decoder blocks dominates at both resolutions:
- **518x518**: 8,908 / 14,653 decoder = **60.7%** of decoder FLOPs, **37.9%** of total
- **1080x1920**: 547,222 / 599,711 decoder = **91.3%** of decoder FLOPs, **72.5%** of total

### Head Decoders Are Minor Cost

Since heads process per-view sequences (not cross-view), their FLOPs are small:
- **518x518**: 659 GF / 23,496 GF = **2.8%** of total
- **1080x1920**: 11,384 GF / 754,803 GF = **1.5%** of total

### Bottleneck Hierarchy (1080x1920, sparsity=0)

1. **Cross-view attention (odd backbone)**: 547.2T -- **72.5%** of total
2. **DINOv2 encoder**: 143.4T -- **19.0%** of total
3. **Cross-view MLP (odd backbone)**: 26.0T -- **3.4%** of total
4. **Per-view decoder blocks (even backbone)**: 13.4T -- **1.8%** of total
5. **Prediction heads (per-view)**: 11.4T -- **1.5%** of total
6. **Output projections**: 0.3T -- **0.0%** of total

---

*Analysis based on config `reliev3r_quant.json` with `decoder_size="large"` (0.95B model), Pi3 depth_aligner, B=1, V=8 views. All FLOPs are theoretical (2 FLOPs per MAC, no accounting for kernel fusion, flash attention IO savings, or activation checkpointing). Sparsity model reflects actual `FlashAttentionRope` implementation: K,V subsampled, Q stays at full N, output stays at (B, N, C).*