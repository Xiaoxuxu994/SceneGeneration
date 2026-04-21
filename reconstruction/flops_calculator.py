# =====================================================
# Reliev3R FLOPs Analysis (Corrected)
# =====================================================
# Config: depth_aligner.pi3, decoder_size='large' (0.95B)
# B=1, V=N=8 views
#
# Corrections vs previous version:
# 1. Sparsity formula: FlashAttentionRope.sparsity subsamples only K,V (not Q),
#    so attention cost = 4*n*(n/s)*d, NOT 4*(n/s)*(n/s)*d
# 2. Head decoders process per-view (batch=B*V, seq=n_pv), NOT cross-view (B, V*n_pv)
# 3. Head decoder blocks have no qk_norm and no LayerScale (init_values=None)
# 4. Sparsity convention matches code: 0=original (FlashAttentionRope.sparsity=0)

# ---------- Architecture parameters ----------
V = 8  # number of views
d_enc = 1024   # DINOv2-L embed_dim
enc_depth = 24  # DINOv2-L blocks
enc_heads = 16
mlp_ratio = 4
patch_size = 14
n_reg_enc = 4   # DINOv2 reg tokens
cls_token = 1   # CLS token in DINOv2

d_dec = 1024    # decoder dim (model_size 0.95B)
dec_depth = 36  # decoder blocks
dec_heads = 16
n_reg_dec = 5   # Pi3 decoder register tokens

head_depth = 5  # TransformerDecoder depth for each head
d_head = 1024   # head internal dim (except camera_out=512)
d_head_in = 2 * d_dec  # head input dim (concatenated decoder outputs)

# ---------- Resolutions ----------
res1 = (518, 518)
res2 = (1080, 1920)

def compute_flops(H, W, label):
    ph = H // patch_size  # integer division
    pw = W // patch_size
    # Adjust to exact divisibility (like the code does via padding)
    if H % patch_size != 0:
        ph = (H + patch_size - 1) // patch_size
    if W % patch_size != 0:
        pw = (W + patch_size - 1) // patch_size

    n_patches = ph * pw

    # ---- Encoder tokens (per image) ----
    n_enc = n_patches + cls_token + n_reg_enc  # patches + cls + registers

    # ---- Decoder tokens (per view after concat register) ----
    n_pv = n_patches + n_reg_dec  # per-view: patches + 5 registers
    n_cv = V * n_pv               # cross-view: all views concatenated

    print(f'\n===== {label}: {H}x{W} =====')
    print(f'Patch grid: {ph}x{pw} = {n_patches} patches')
    print(f'Encoder tokens per image: {n_enc}')
    print(f'Decoder per-view tokens: {n_pv}')
    print(f'Decoder cross-view tokens: {n_cv}')

    # =========================================
    # Stage 1: Encoder (DINOv2-L/14) — per image, run V times
    # =========================================
    flops_patch_embed = 2 * patch_size * patch_size * 3 * d_enc * n_patches

    def vit_block_flops(n, d, h_count, mlp_r):
        norm1 = 4 * n * d
        qkv = 2 * n * d * 3 * d
        attn = 4 * n * n * d
        out_proj = 2 * n * d * d
        ls1 = 2 * n * d  # LayerScale: element-wise
        norm2 = 4 * n * d
        hidden = int(d * mlp_r)
        mlp_fc1 = 2 * n * d * hidden
        mlp_fc2 = 2 * n * hidden * d
        ls2 = 2 * n * d
        total = norm1 + qkv + attn + out_proj + ls1 + norm2 + mlp_fc1 + mlp_fc2 + ls2
        return total, {
            'norm1': norm1, 'qkv_proj': qkv, 'attn': attn, 'out_proj': out_proj, 'ls1': ls1,
            'norm2': norm2, 'mlp_fc1': mlp_fc1, 'mlp_fc2': mlp_fc2, 'ls2': ls2
        }

    enc_block_total, enc_block_detail = vit_block_flops(n_enc, d_enc, enc_heads, mlp_ratio)
    enc_blocks_total = enc_depth * enc_block_total
    enc_final_norm = 4 * n_enc * d_enc

    enc_total_per_image = flops_patch_embed + enc_blocks_total + enc_final_norm
    enc_total = V * enc_total_per_image  # V images

    print(f'\n--- Stage 1: DINOv2-L/14 Encoder (x{V} images) ---')
    print(f'PatchEmbed Conv:  {flops_patch_embed/1e9:.4f} GFLOPs per image')
    print(f'Per block total:  {enc_block_total/1e9:.4f} GFLOPs')
    print(f'  norm1:     {enc_block_detail["norm1"]/1e6:.2f} MFLOPs')
    print(f'  qkv_proj:  {enc_block_detail["qkv_proj"]/1e6:.2f} MFLOPs')
    print(f'  attn:      {enc_block_detail["attn"]/1e6:.2f} MFLOPs')
    print(f'  out_proj:  {enc_block_detail["out_proj"]/1e6:.2f} MFLOPs')
    print(f'  ls1:       {enc_block_detail["ls1"]/1e6:.2f} MFLOPs')
    print(f'  norm2:     {enc_block_detail["norm2"]/1e6:.2f} MFLOPs')
    print(f'  mlp_fc1:   {enc_block_detail["mlp_fc1"]/1e6:.2f} MFLOPs')
    print(f'  mlp_fc2:   {enc_block_detail["mlp_fc2"]/1e6:.2f} MFLOPs')
    print(f'  ls2:       {enc_block_detail["ls2"]/1e6:.2f} MFLOPs')
    print(f'{enc_depth} blocks total: {enc_blocks_total/1e9:.4f} GFLOPs per image')
    print(f'Final norm:       {enc_final_norm/1e6:.2f} MFLOPs')
    print(f'Encoder total (V={V}): {enc_total/1e9:.4f} GFLOPs')

    # =========================================
    # Stage 2: Projection — Identity (0 FLOPs for 0.95B)
    # =========================================
    proj_flops = 0
    proj_norm = 0  # Identity too since enc_dim == dec_dim
    print(f'\n--- Stage 2: Projection ---')
    print(f'Identity projection (enc_dim={d_enc} == dec_dim={d_dec}): 0 GFLOPs')

    # =========================================
    # Stage 3: Decoder (36 BlockRope) — Pi3 backbone
    # =========================================
    # Even blocks (i%2==0): per-view self-attention
    #   shape (B*V, n_pv, d), attention on n=n_pv tokens
    # Odd blocks (i%2==1): cross-view attention
    #   shape (B, V*n_pv, d), attention on n=V*n_pv tokens
    #
    # Backbone blocks have: qk_norm=True, LayerScale (init_values=0.01)

    def backbone_block_flops(n, d, h_count, mlp_r, sparsity=0):
        """
        BlockRope FLOPs for Pi3 backbone decoder blocks.
        Has qk_norm and LayerScale (init_values=0.01).

        sparsity: matches FlashAttentionRope.sparsity
          0 = no sparsity (original)
          N = subsample KV to N//sparsity tokens (Q stays full)
        """
        hidden = int(d * mlp_r)

        norm1 = 4 * n * d
        qkv = 2 * n * d * 3 * d

        # Sparsity: FlashAttentionRope subsamples k,v to n_kv = n // sparsity
        # Q stays at n tokens → attention is (n, n_kv) not (n_kv, n_kv)
        # Output shape stays (B, n, C)
        if sparsity <= 1:  # 0 or 1 → dense
            attn = 4 * n * n * d
        else:
            n_kv = n // sparsity
            attn = 4 * n * n_kv * d  # QK^T: 2*n*n_kv*d + attn@V: 2*n*n_kv*d

        qk_norm_cost = 4 * n * d  # q_norm + k_norm (LayerNorm on head_dim)
        rope_cost = 4 * n * d  # rotary emb

        out_proj = 2 * n * d * d
        ls1 = 2 * n * d  # LayerScale (init_values=0.01, not Identity)

        norm2 = 4 * n * d
        mlp_fc1 = 2 * n * d * hidden
        mlp_fc2 = 2 * n * hidden * d
        ls2 = 2 * n * d  # LayerScale

        total = norm1 + qkv + attn + qk_norm_cost + rope_cost + out_proj + ls1 + norm2 + mlp_fc1 + mlp_fc2 + ls2
        return total, {
            'norm1': norm1, 'qkv_proj': qkv, 'attn': attn, 'qk_norm': qk_norm_cost,
            'rope': rope_cost, 'out_proj': out_proj, 'ls1': ls1,
            'norm2': norm2, 'mlp_fc1': mlp_fc1, 'mlp_fc2': mlp_fc2, 'ls2': ls2
        }

    # Even blocks: per-view (n = n_pv)
    pv_block_flops, pv_detail = backbone_block_flops(n_pv, d_dec, dec_heads, mlp_ratio)
    # Odd blocks: cross-view (n = n_cv = V*n_pv)
    cv_block_flops_0, cv_detail_0 = backbone_block_flops(n_cv, d_dec, dec_heads, mlp_ratio)

    n_even = (dec_depth + 1) // 2   # blocks 0,2,4,...,34 → 18 even blocks
    n_odd = dec_depth // 2           # blocks 1,3,5,...,35 → 18 odd blocks

    dec_even_total = n_even * pv_block_flops
    dec_odd_total_0 = n_odd * cv_block_flops_0
    dec_total_0 = dec_even_total + dec_odd_total_0

    print(f'\n--- Stage 3: {dec_depth}-block Pi3 Backbone Decoder ---')
    print(f'Even blocks (per-view, n={n_pv}): {n_even} blocks')
    print(f'  Per block: {pv_block_flops/1e9:.4f} GFLOPs')
    print(f'  detail: norm1={pv_detail["norm1"]/1e6:.2f}M, qkv={pv_detail["qkv_proj"]/1e6:.2f}M, attn={pv_detail["attn"]/1e6:.2f}M, qk_norm={pv_detail["qk_norm"]/1e6:.2f}M, rope={pv_detail["rope"]/1e6:.2f}M, out_proj={pv_detail["out_proj"]/1e6:.2f}M, ls1={pv_detail["ls1"]/1e6:.2f}M')
    print(f'  detail: norm2={pv_detail["norm2"]/1e6:.2f}M, mlp_fc1={pv_detail["mlp_fc1"]/1e6:.2f}M, mlp_fc2={pv_detail["mlp_fc2"]/1e6:.2f}M, ls2={pv_detail["ls2"]/1e6:.2f}M')
    print(f'  Even total: {dec_even_total/1e9:.4f} GFLOPs')
    print(f'Odd blocks (cross-view, n={n_cv}): {n_odd} blocks')
    print(f'  Per block (sparsity=0): {cv_block_flops_0/1e9:.4f} GFLOPs')
    print(f'  detail: norm1={cv_detail_0["norm1"]/1e6:.2f}M, qkv={cv_detail_0["qkv_proj"]/1e6:.2f}M, attn={cv_detail_0["attn"]/1e6:.2f}M, qk_norm={cv_detail_0["qk_norm"]/1e6:.2f}M, rope={cv_detail_0["rope"]/1e6:.2f}M, out_proj={cv_detail_0["out_proj"]/1e6:.2f}M, ls1={cv_detail_0["ls1"]/1e6:.2f}M')
    print(f'  detail: norm2={cv_detail_0["norm2"]/1e6:.2f}M, mlp_fc1={cv_detail_0["mlp_fc1"]/1e6:.2f}M, mlp_fc2={cv_detail_0["mlp_fc2"]/1e6:.2f}M, ls2={cv_detail_0["ls2"]/1e6:.2f}M')
    print(f'  Odd total (sparsity=0): {dec_odd_total_0/1e9:.4f} GFLOPs')
    print(f'Backbone decoder total (sparsity=0): {dec_total_0/1e9:.4f} GFLOPs')

    # =========================================
    # Stage 4: Prediction Heads (per-view processing)
    # =========================================
    # Each head: TransformerDecoder(in_dim=2*d_dec, dec_embed_dim=1024, depth=5, out_dim=...)
    # Input: hidden from decoder, shape (B*V, n_pv, 2*d_dec)
    # After projection: (B*V, n_pv, 1024) — per-view (batch=B*V)
    # 5 BlockRope blocks on n=n_pv tokens per batch element
    #
    # Head blocks have: qk_norm=False, no LayerScale (init_values=None → nn.Identity)

    def head_block_flops(n, d, h_count, mlp_r, sparsity=0):
        """
        BlockRope FLOPs for head decoder blocks.
        No qk_norm, no LayerScale (init_values=None → Identity).

        sparsity: matches FlashAttentionRope.sparsity
          0 = no sparsity (original)
          N = subsample KV to N//sparsity tokens (Q stays full)
        """
        hidden = int(d * mlp_r)

        norm1 = 4 * n * d
        qkv = 2 * n * d * 3 * d

        if sparsity <= 1:
            attn = 4 * n * n * d
        else:
            n_kv = n // sparsity
            attn = 4 * n * n_kv * d

        # No qk_norm (nn.Identity)
        # RoPE still applied (rope parameter passed to FlashAttentionRope)
        rope_cost = 4 * n * d

        out_proj = 2 * n * d * d
        # No LayerScale (nn.Identity, 0 FLOPs)

        norm2 = 4 * n * d
        mlp_fc1 = 2 * n * d * hidden
        mlp_fc2 = 2 * n * hidden * d
        # No LayerScale

        total = norm1 + qkv + attn + rope_cost + out_proj + norm2 + mlp_fc1 + mlp_fc2
        return total, {
            'norm1': norm1, 'qkv_proj': qkv, 'attn': attn, 'rope': rope_cost,
            'out_proj': out_proj, 'norm2': norm2, 'mlp_fc1': mlp_fc1, 'mlp_fc2': mlp_fc2
        }

    def head_flops(n_pv, d_in, d_internal, d_out, depth, sparsity=0):
        """FLOPs for one TransformerDecoder head (per-view processing)."""
        proj = 2 * n_pv * d_in * d_internal
        blk_f, blk_d = head_block_flops(n_pv, d_internal, dec_heads, mlp_ratio, sparsity=sparsity)
        blocks_total = depth * blk_f
        linear_out = 2 * n_pv * d_internal * d_out
        total = proj + blocks_total + linear_out
        return total, {'proj': proj, 'blocks': blocks_total, 'linear_out': linear_out, 'blk_f': blk_f, 'blk_d': blk_d}

    # sparsity=0 (original) for heads
    depth_dec_flops, depth_detail = head_flops(n_pv, d_head_in, 1024, 1024, head_depth)
    conf_dec_flops, conf_detail = head_flops(n_pv, d_head_in, 1024, 1024, head_depth)
    cam_dec_flops, cam_detail = head_flops(n_pv, d_head_in, 1024, 512, head_depth)

    heads_total = depth_dec_flops + conf_dec_flops + cam_dec_flops

    print(f'\n--- Stage 4: Prediction Heads (per-view, n={n_pv}, batch=B*V={V}) ---')
    print(f'Note: Heads process per-view (NOT cross-view). Batch dimension is B*V={V}.')
    print(f'Per head block (no qk_norm, no LayerScale):')
    blk_d = depth_detail['blk_d']
    print(f'  norm1={blk_d["norm1"]/1e6:.2f}M, qkv={blk_d["qkv_proj"]/1e6:.2f}M, attn={blk_d["attn"]/1e6:.2f}M, rope={blk_d["rope"]/1e6:.2f}M, out_proj={blk_d["out_proj"]/1e6:.2f}M')
    print(f'  norm2={blk_d["norm2"]/1e6:.2f}M, mlp_fc1={blk_d["mlp_fc1"]/1e6:.2f}M, mlp_fc2={blk_d["mlp_fc2"]/1e6:.2f}M')
    print(f'  Per block total: {depth_detail["blk_f"]/1e9:.4f} GFLOPs')
    print(f'Depth decoder:  proj={depth_detail["proj"]/1e9:.4f}G, {head_depth} blocks={depth_detail["blocks"]/1e9:.4f}G, linear_out={depth_detail["linear_out"]/1e9:.4f}G → total={depth_dec_flops/1e9:.4f}G')
    print(f'Conf decoder:   proj={conf_detail["proj"]/1e9:.4f}G, {head_depth} blocks={conf_detail["blocks"]/1e9:.4f}G, linear_out={conf_detail["linear_out"]/1e9:.4f}G → total={conf_dec_flops/1e9:.4f}G')
    print(f'Camera decoder: proj={cam_detail["proj"]/1e9:.4f}G, {head_depth} blocks={cam_detail["blocks"]/1e9:.4f}G, linear_out={cam_detail["linear_out"]/1e9:.4f}G → total={cam_dec_flops/1e9:.4f}G')
    print(f'Heads total: {heads_total/1e9:.4f} GFLOPs')

    # =========================================
    # Stage 5: Output projections
    # =========================================
    n_ptokens = n_patches  # patch tokens only (register excluded via patch_start_idx)

    depth_head_proj = 2 * V * n_ptokens * 1024 * (1 * patch_size**2)
    conf_head_proj = 2 * V * n_ptokens * 1024 * (1 * patch_size**2)

    cam_resconv = 2 * 3 * 2 * V * n_ptokens * 512 * 512
    cam_mlps = 2 * V * (2 * 512 * 512 + 2 * 512 * 512)
    cam_fc = 2 * V * (512 * 3 + 512 * 9 + 512 * 2)

    stage5 = depth_head_proj + conf_head_proj + cam_resconv + cam_mlps + cam_fc

    print(f'\n--- Stage 5: Output Projections ---')
    print(f'Depth head Linear:   {depth_head_proj/1e9:.4f} GFLOPs')
    print(f'Conf head Linear:    {conf_head_proj/1e9:.4f} GFLOPs')
    print(f'Camera ResConv+MLP:  {(cam_resconv+cam_mlps+cam_fc)/1e9:.4f} GFLOPs')
    print(f'Stage 5 total: {stage5/1e9:.4f} GFLOPs')

    # =========================================
    # Grand Total (sparsity=0, original)
    # =========================================
    grand_total_0 = enc_total + proj_flops + proj_norm + dec_total_0 + heads_total + stage5
    print(f'\n===== GRAND TOTAL (sparsity=0, original) =====')
    print(f'Encoder:      {enc_total/1e9:.2f} GFLOPs ({enc_total/grand_total_0*100:.1f}%)')
    print(f'Projection:   {(proj_flops+proj_norm)/1e9:.2f} GFLOPs')
    print(f'Backbone Decoder: {dec_total_0/1e9:.2f} GFLOPs ({dec_total_0/grand_total_0*100:.1f}%)')
    print(f'  - Even (per-view):   {dec_even_total/1e9:.2f} GFLOPs ({dec_even_total/grand_total_0*100:.1f}%)')
    print(f'  - Odd (cross-view):  {dec_odd_total_0/1e9:.2f} GFLOPs ({dec_odd_total_0/grand_total_0*100:.1f}%)')
    print(f'Prediction Heads:  {heads_total/1e9:.2f} GFLOPs ({heads_total/grand_total_0*100:.1f}%)')
    print(f'Output proj:   {stage5/1e9:.2f} GFLOPs ({stage5/grand_total_0*100:.1f}%)')
    print(f'TOTAL:         {grand_total_0/1e9:.2f} GFLOPs')

    # =========================================
    # Cross-view sparsity comparison (Pi3 backbone odd blocks only)
    # =========================================
    print(f'\n===== Cross-View Attention Sparsity (Pi3 Backbone Odd Blocks, n={n_cv}) =====')
    print(f'FlashAttentionRope.sparsity subsamples K,V only (Q stays at N={n_cv})')
    print(f'{"sparsity":>10} {"n_kv":>8} {"attn/blk(GF)":>14} {"ratio":>8} {"odd_total(GF)":>14} {"dec_total(GF)":>14} {"grand(GF)":>14} {"savings":>10}')

    # Store for comparison
    cv_results = {}
    for s in [0, 2, 4, 8]:
        cv_f, cv_d = backbone_block_flops(n_cv, d_dec, dec_heads, mlp_ratio, sparsity=s)
        odd_total_s = n_odd * cv_f
        dec_total_s = dec_even_total + odd_total_s
        grand_s = enc_total + proj_flops + proj_norm + dec_total_s + heads_total + stage5
        ratio = cv_d['attn'] / cv_detail_0['attn'] if cv_detail_0['attn'] > 0 else 1.0
        savings = (1 - grand_s / grand_total_0) * 100
        n_kv = n_cv if s <= 1 else n_cv // s
        cv_results[s] = {'grand': grand_s, 'odd_total': odd_total_s, 'dec_total': dec_total_s, 'attn_per_blk': cv_d['attn']}
        print(f'{s:>10d} {n_kv:>8d} {cv_d["attn"]/1e9:>14.4f} {ratio:>8.3f} {odd_total_s/1e9:>14.2f} {dec_total_s/1e9:>14.2f} {grand_s/1e9:>14.2f} {savings:>+9.1f}%')

    # =========================================
    # Per-view sparsity comparison (even blocks + heads)
    # =========================================
    print(f'\n===== Per-View Attention Sparsity (n={n_pv}, Even Backbone + Head Blocks) =====')
    print(f'Note: Savings are modest since per-view sequences are short ({n_pv} tokens)')

    # Backbone even blocks with per-view sparsity
    pv_backbone_savings = {}
    for s in [0, 2, 4, 8]:
        pv_f, pv_d = backbone_block_flops(n_pv, d_dec, dec_heads, mlp_ratio, sparsity=s)
        pv_backbone_savings[s] = n_even * pv_f

    # Head blocks with per-view sparsity
    pv_head_savings = {}
    for s in [0, 2, 4, 8]:
        # All 3 heads × 5 blocks each = 15 head blocks (per-view, no qk_norm/ls)
        total_head_s = 0
        for head_d_out, n_heads_same in [(1024, 2), (512, 1)]:  # depth+conf=1024, camera=512
            h_flops, _ = head_flops(n_pv, d_head_in, 1024, head_d_out, head_depth, sparsity=s)
            total_head_s += n_heads_same * h_flops
        pv_head_savings[s] = total_head_s

    print(f'{"sparsity":>10} {"even_backbone(GF)":>18} {"heads(GF)":>12} {"grand(GF)":>14} {"savings":>10}')
    for s in [0, 2, 4, 8]:
        even_total_s = pv_backbone_savings[s]
        heads_total_s = pv_head_savings[s]
        # Reconstruct grand total: use cross-view sparsity=0 but vary per-view sparsity
        # Actually sparsity is a class-level attribute, affects ALL blocks globally.
        # Show two scenarios: (a) sparsity on cross-view only, (b) sparsity everywhere
        pass

    # Combined scenario: sparsity applied to ALL blocks (class-level attribute)
    print(f'\n===== Combined Sparsity (Class-Level, Affects ALL FlashAttentionRope Instances) =====')
    print(f'FlashAttentionRope.sparsity is a class attribute — setting it affects all blocks.')
    print(f'{"sparsity":>10} {"attn_cv/blk(GF)":>16} {"attn_pv/blk(GF)":>16} {"grand(GF)":>14} {"savings":>10}')

    for s in [0, 2, 4, 8]:
        # Cross-view (odd backbone)
        cv_f_s, cv_d_s = backbone_block_flops(n_cv, d_dec, dec_heads, mlp_ratio, sparsity=s)
        odd_total_s = n_odd * cv_f_s

        # Per-view (even backbone)
        pv_f_s, pv_d_s = backbone_block_flops(n_pv, d_dec, dec_heads, mlp_ratio, sparsity=s)
        even_total_s = n_even * pv_f_s

        # Heads (per-view, no qk_norm/ls)
        d_h_f, _ = head_flops(n_pv, d_head_in, 1024, 1024, head_depth, sparsity=s)
        c_h_f, _ = head_flops(n_pv, d_head_in, 1024, 1024, head_depth, sparsity=s)
        cam_h_f, _ = head_flops(n_pv, d_head_in, 1024, 512, head_depth, sparsity=s)
        heads_total_s = d_h_f + c_h_f + cam_h_f

        dec_total_s = even_total_s + odd_total_s
        grand_s = enc_total + proj_flops + proj_norm + dec_total_s + heads_total_s + stage5
        savings = (1 - grand_s / grand_total_0) * 100
        print(f'{s:>10d} {cv_d_s["attn"]/1e9:>16.4f} {pv_d_s["attn"]/1e9:>16.4f} {grand_s/1e9:>14.2f} {savings:>+9.1f}%')

    # Per-block attention cost comparison
    print(f'\n===== Attention FLOPs Per Block (sparsity comparison) =====')
    print(f'Corrected formula: 4*N*(N/s)*d (K,V subsampled to N/s, Q stays at N)')
    print(f'{"":>25} {"sparsity=0":>12} {"sparsity=2":>12} {"sparsity=4":>12} {"sparsity=8":>12}')

    # Cross-view backbone blocks
    print(f'{"Cross-view (n="+str(n_cv)+")":>25}', end='')
    for s in [0, 2, 4, 8]:
        _, d = backbone_block_flops(n_cv, d_dec, dec_heads, mlp_ratio, sparsity=s)
        print(f'{d["attn"]/1e9:>12.4f}', end='')
    print(f'  (GFLOPs)')

    # Per-view backbone blocks
    print(f'{"Per-view (n="+str(n_pv)+")":>25}', end='')
    for s in [0, 2, 4, 8]:
        _, d = backbone_block_flops(n_pv, d_dec, dec_heads, mlp_ratio, sparsity=s)
        print(f'{d["attn"]/1e9:>12.4f}', end='')
    print(f'  (GFLOPs)')

    # Per-view head blocks
    print(f'{"Head per-view (n="+str(n_pv)+")":>25}', end='')
    for s in [0, 2, 4, 8]:
        _, d = head_block_flops(n_pv, 1024, dec_heads, mlp_ratio, sparsity=s)
        print(f'{d["attn"]/1e9:>12.4f}', end='')
    print(f'  (GFLOPs)')

    # Ratio vs original
    print(f'\n{"Attention ratio vs sparsity=0":>25} {"":>12} {"sparsity=2":>12} {"sparsity=4":>12} {"sparsity=8":>12}')
    for name, n_val, block_fn in [
        ('Cross-view', n_cv, lambda n, s: backbone_block_flops(n, d_dec, dec_heads, mlp_ratio, sparsity=s)),
        ('Per-view', n_pv, lambda n, s: backbone_block_flops(n, d_dec, dec_heads, mlp_ratio, sparsity=s)),
        ('Head', n_pv, lambda n, s: head_block_flops(n, 1024, dec_heads, mlp_ratio, sparsity=s)),
    ]:
        _, d0 = block_fn(n_val, 0)
        print(f'{name+" (n="+str(n_val)+")":>25} {"1.000":>12}', end='')
        for s in [2, 4, 8]:
            _, ds = block_fn(n_val, s)
            print(f'{ds["attn"]/d0["attn"]:>12.3f}', end='')
        print()

# Run for both resolutions
compute_flops(518, 518, 'Resolution 1')
compute_flops(1080, 1920, 'Resolution 2')