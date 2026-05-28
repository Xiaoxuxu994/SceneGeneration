class Pi3X(nn.Module):
    def __init__(
            self,
            ckpts=None,    
            use_checkpoint=False,
            checkpoint_strategy=None,
            weight_source='vggt',
            use_multimodal=True
        ):
        super().__init__()

        self.use_multimodal = use_multimodal
        self.use_checkpoint = use_checkpoint
        self.checkpoint_strategy = checkpoint_strategy

        # ----------------------
        #        Encoder
        # ----------------------
        from models.moge.model.dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg, dinov2_vits14_reg, dinov2_vitb14_reg
        self.encoder = dinov2_vitl14_reg(pretrained=False, use_checkpoint=True)             # always use checkpoint
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        freq = 100
        self.rope = RoPE2D(freq=freq)
        self.position_getter = PositionGetter()


        # ----------------------
        #        Decoder
        # ----------------------
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36                      #### there is a bug, in fact we use 18 layers, it should be 24 originally
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=self.rope
        ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #    Loading Weight
        # ----------------------
        if ckpts is None:
            if weight_source == 'vggt':
                # vggt_weight = torch.load('ckpts/vggt.pt', weights_only=False)
                vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
                vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
                print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

                vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
                vggt_dec_weight1 = {}
                for k in list(vggt_dec_weight.keys()):
                    idx = k.split('.')[0]
                    other = k[len(idx):]
                    vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
                vggt_dec_weight = vggt_dec_weight1 

                vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
                for k in list(vggt_dec_weight_frame.keys()):
                    idx = k.split('.')[0]
                    other = k[len(idx):]
                    vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

                print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

                del vggt_weight
            elif weight_source == 'dino':
                ckpt_dinov2 = torch.load('ckpts/dinov2_vitl14_reg4_pretrain.pth', weights_only=False, map_location=torch.device('cpu'))
                print("Loading dinov2 encoder", self.encoder.load_state_dict(ckpt_dinov2, strict=False))

                # load dinov2 weight for decoder
                dinov2_blocks_weight = {k[7:]: ckpt_dinov2[k] for k in list(ckpt_dinov2.keys()) if k.startswith('blocks') }
                print("Loading dinov2 decoder", self.decoder.load_state_dict(dinov2_blocks_weight, strict=False))

                del ckpt_dinov2
            elif weight_source == 'pi3':
                ckpt_pi3 = load_file('ckpts/Pi3/model.safetensors')
                pi3_enc_weight = {k.replace('encoder.', ''):ckpt_pi3[k] for k in list(ckpt_pi3.keys()) if k.startswith('encoder.')}
                print("Loading pi3 encoder", self.encoder.load_state_dict(pi3_enc_weight, strict=False))

                # load dinov2 weight for decoder
                pi3_dec_weight = {k.replace('decoder.', ''):ckpt_pi3[k] for k in list(ckpt_pi3.keys()) if k.startswith('decoder.')}
                print("Loading pi3 decoder", self.decoder.load_state_dict(pi3_dec_weight, strict=False))

                del ckpt_pi3


        # -----------------------
        #       multi-modal
        # -----------------------
        if use_multimodal:
            ## Depth encoder
            self.depth_encoder = deepcopy(self.encoder)
            del self.depth_encoder.patch_embed
            self.depth_encoder.patch_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            self.depth_emb = nn.Parameter(torch.zeros(1, 1, 1024))

            ## Ray embedding
            self.ray_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            nn.init.constant_(self.ray_embed.proj.weight, 0)
            nn.init.constant_(self.ray_embed.proj.bias, 0)

            ## Pose inject blocks
            self.pose_inject_blk = nn.ModuleList([PoseInjectBlock(
                dim=1024,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
            ) for _ in range(5)])


        # ------------------------------
        #           Head
        # ------------------------------
        ## --------------- Point ---------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=use_checkpoint
        )
        # self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.point_head = ConvHead(
                num_features=4, 
                dim_in=dec_embed_dim,
                # projects=nn.Linear(1024, 1024),
                projects=nn.Identity(),
                dim_out=[2, 1], 
                dim_proj=1024,
                dim_upsample=[256, 128, 64],
                dim_times_res_block_hidden=2,
                num_res_blocks=2,
                res_block_norm='group_norm',
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                using_uv=True
            )

        ## --------------- Camera ---------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=use_checkpoint
        )
        self.camera_head = CameraHead(dim=512)

        # ## --------------- Flow ---------------
        # omega = 2 * torch.pi * torch.randn(dec_embed_dim, 2)
        # self.omega = nn.Buffer(omega)

        # self.tanh = nn.Tanh()
        # # regression
        # self.flow_decoder = ContextTransformerDecoder(
        #     in_dim=1024*2,
        #     dec_embed_dim=1024,
        #     dec_num_heads=16,
        #     out_dim=1024,
        #     rope=self.rope,
        #     use_checkpoint=use_checkpoint
        # )

        # self.flow_head = LinearPts3d(14, 1024, 3)

        ## --------------- Metric ---------------
        self.metric_token = nn.Parameter(torch.randn(1, 1, 2*self.dec_embed_dim))
        self.metric_decoder = ContextOnlyTransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=512,
            dec_num_heads=8,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=use_checkpoint
        )
        self.metric_head = nn.Linear(512, 1)
        nn.init.normal_(self.metric_token, std=1e-6)


        ## -------------- Conf ------------------
        self.conf_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=use_checkpoint
        )
        self.conf_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            # projects=nn.Linear(1024, 1024),
            projects=nn.Identity(),
            dim_out=[1], 
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )


        # ------------------------------
        #      Loading checkpoint
        # ------------------------------
        if ckpts is not None:
            checkpoint = torch.load(ckpts, weights_only=False, map_location='cpu')

            # for name, param in self.named_parameters():
            #     if 'conf' not in name:
            #         param.requires_grad = False

            # # ===================

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Pi3X] Load checkpoints from {ckpts}: {res}')


            del checkpoint
            torch.cuda.empty_cache()

    def forward(
        self,
        imgs,
        order_flow=None,
        depths=None,
        intrinsics=None,
        poses=None,
        with_prior=None,
        **kargs
    ):
        # device = imgs.device
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # encode
        hidden, poses_, use_depth_mask, use_pose_mask, norm_factor = self.encode(imgs, with_prior, depths, intrinsics, poses, **kargs)
        hidden = hidden.reshape(B, N, -1, self.dec_embed_dim)

        # decode
        hidden, pos, ref_idxs = self.decode(hidden, N, H, W, poses_, use_pose_mask)

        # # head
        outputs = self.forward_head(hidden, pos, order_flow, B, N, H, W, patch_h, patch_w)

        outputs.update(dict(order_flow=order_flow, use_depth_mask=use_depth_mask, use_pose_mask=use_pose_mask, ref_idxs=ref_idxs))

        return outputs
    
    def encode(
        self, 
        imgs, 
        with_prior,
        depths=None,
        intrinsics=None,
        poses=None,
        **kargs
    ):
        B, N, _, H, W = imgs.shape
        device = imgs.device

        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)["x_norm_patchtokens"]

        if self.use_multimodal:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                if with_prior is None:
                    p_depth = 0.3
                    p_ray = 0.5
                    p_pose = 0.2
                elif with_prior is True:
                    p_depth = p_ray = p_pose = 1.0
                else:
                    p_depth = p_ray = p_pose = 0.0

                if depths is None:
                    p_depth = 0.0
                    depths = torch.zeros((B, N, H, W), device=imgs.device)

                if intrinsics is None:
                    p_ray = 0.0
                    rays = torch.zeros((B, N, H, W, 2), device=imgs.device)
                else:
                    pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(device).float()[None].repeat(B, 1, 1, 1)
                    rays = torch.einsum('bnij, bhwj -> bnhwi', torch.inverse(intrinsics), pix)[..., :2]
                    # rays = F.normalize(rays, dim=-1).reshape(B, N, H, W, 3)                   # don't normalize, so the pred['xy'] is the same as input rays

                if poses is None:
                    p_pose = 0.0
                    poses = torch.eye(4, device=device)[None, None].repeat(B, N, 1, 1)
                else:
                    assert rays is not None                     # rays should be along with poses
                    
                mask_add_depth = torch.rand((B, N), device=device) <= p_depth
                mask_add_ray = torch.rand((B, N), device=device) <= p_ray
                mask_add_pose = torch.rand((B, N), device=device) <= p_pose

                # pose is injected relatively. so at least two frame should be true.
                num_valid_pose = mask_add_pose.sum(dim=1)
                bad_indices = (num_valid_pose == 1)
                mask_add_pose[bad_indices] = False

                if 'dataset_name' in kargs:
                    teach_batch = [i for i in range(B) if kargs['dataset_name'][i] in __TEACH_DATASETS__]
                    mask_add_depth[teach_batch, :] = False

                # normalize depth and pose
                normalized_depths, dep_median = self.normalize_depth(depths, method='mean')
                scale_aug = 0.8 + torch.rand((B,), device=device) * 0.4
                normalized_depths /= scale_aug.view(B, 1, 1, 1)
                dep_median *= scale_aug

                depths_masks = (normalized_depths > 0).float()
                depths_masks = depths_masks.reshape(B*N, 1, H, W)

                poses_ = torch.einsum('bij, bnjk -> bnik', se3_inverse(poses[:, 0]), poses)
                poses_[..., :3, 3] /= dep_median.view(B, 1, 1)

                # noramlize for the batch not using depth
                use_depth_batch_mask = mask_add_depth.sum(dim=1) > 0
                if (~use_depth_batch_mask).sum() > 0 and N > 1:
                    pose_scale = poses_[..., 1:, :3, 3].norm(dim=-1)

                    static_threshold = 2e-2
                    is_static_mask = pose_scale.max(dim=1)[0] < static_threshold

                    pose_scale = pose_scale.mean(dim=1)
                    scale_aug = 0.8 + torch.rand((B,), device=device) * 0.4
                    pose_scale *= scale_aug

                    final_moving_mask = torch.logical_and(~use_depth_batch_mask, ~is_static_mask)
                    poses_[final_moving_mask, ..., :3, 3] /= (pose_scale.view(B, 1, 1)[final_moving_mask] + 1e-8)
                    normalized_depths[final_moving_mask] /= (pose_scale.view(B, 1, 1, 1)[final_moving_mask] + 1e-8)

                    dep_median[final_moving_mask] *= pose_scale[final_moving_mask]

                if with_prior is None:
                    add_noise_batch = torch.rand(B) > 0.5
                else:
                    add_noise_batch = torch.rand(B) > 1   

                if N > 1:
                    poses_[add_noise_batch, 1:] = add_randomized_smooth_pose_noise_torch(poses_[add_noise_batch, 1:])

                normalized_depths = normalized_depths.reshape(B*N, 1, H, W)

            depth_emb = self.depth_encoder(torch.cat([normalized_depths, depths_masks], dim=1), is_training=True)["x_norm_patchtokens"] + self.depth_emb
            ray_emb = self.ray_embed(rays.reshape(B*N, H, W, 2).permute(0, 3, 1, 2))

            use_depth_mask = mask_add_depth
            use_pose_mask = mask_add_pose

            hidden = hidden + ray_emb * mask_add_ray.reshape(B*N, 1, 1)
            hidden = hidden + depth_emb * mask_add_depth.reshape(B*N, 1, 1)

            return hidden, poses_, use_depth_mask, use_pose_mask, dep_median
        
        return hidden, None, None, None, None
    
    def forward_head(self, hidden, pos, order_flow, B, N, H, W, patch_h, patch_w):
        device = hidden.device
        hw = patch_h*patch_w+self.patch_start_idx

        # # decode flow
        # if order_flow is not None:

        #     x = get_normalized_grid(B*N, patch_h, patch_w, overload_device=device)
        #     x_emb = nn.functional.linear(
        #         x.reshape(B*N, patch_h * patch_w, 2), self.omega
        #     )
        #     pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

        #     context = hidden.reshape(B, N, hw, -1)[torch.arange(B)[:, None].to(device), order_flow].reshape(B*N, hw, -1)
        #     context_prefix = context[:, :self.patch_start_idx, :]
        #     context_patches = context[:, self.patch_start_idx:, :]
        #     context_patches_with_pos = context_patches + pos_emb_grid
        #     context_with_pos = torch.cat([context_prefix, context_patches_with_pos], dim=1)

        #     ret_flow = self.flow_decoder(hidden, context_with_pos, xpos=pos, ypos=pos)

        # decode point
        ret_point = self.point_decoder(hidden, xpos=pos)

        # decode camera
        ret_camera = self.camera_decoder(hidden, xpos=pos)

        # decode metric
        pos_hw = pos.reshape(B, N*hw, -1)
        ret_metric = self.metric_decoder(self.metric_token.repeat(B, 1, 1), hidden.reshape(B, N*hw, -1), xpos=pos_hw[:, 0:1], ypos=pos_hw)

        ret_conf = self.conf_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local_points = self.point_head([ret_point[:, self.patch_start_idx:].float()], (H, W)).reshape(B, N, H, W, -1)
            # xy, z = local_points.split([2, 1], dim=-1)

            xy, z = self.point_head(ret_point[:, self.patch_start_idx:].float(), patch_h=patch_h, patch_w=patch_w)
            xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            # z = torch.exp(z)
            z = torch.exp(z.clamp(max=15.0))
            local_points = torch.cat([xy * z, z], dim=-1)

            # if order_flow is not None:
            #     flow = self.flow_head([ret_flow[:, self.patch_start_idx:].float()], (H, W)).reshape(B, N, H, W, -1)
            #     flow, flow_prob = flow.split([2, 1], dim=-1)
            #     flow = self.tanh(flow)
            # else:
            #     flow = flow_prob = None

            camera_poses = self.camera_head(ret_camera[:, self.patch_start_idx:].float(), patch_h, patch_w).reshape(B, N, 4, 4)

            metric = self.metric_head(ret_metric.float()).reshape(B).exp()

            # conf
            conf = self.conf_head(ret_conf[:, self.patch_start_idx:].float(), patch_h=patch_h, patch_w=patch_w)[0]
            conf = conf.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)


        return dict(
            local_points=local_points, xy=xy,
            conf=conf,
            # flow=flow, flow_prob=flow_prob, order_flow=order_flow,
            camera_poses=camera_poses,  
            metric=metric,
        )


    def decode(self, hidden, N, H, W, poses, use_pose_mask):
        device = hidden.device

        if len(hidden.shape) == 4:
            B, N, hw, _ = hidden.shape
        else:
            BN, hw, _ = hidden.shape
            B = BN // N

        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        ref_idxs = None
        hw = hidden.shape[1]
        pose_inject_blk_idx = 0

        pos = self.position_getter(B*N, H//self.patch_size, W//self.patch_size, hidden.device)
        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos_patch = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos_patch], dim=1)

        if self.use_multimodal:
            view_interaction_mask = use_pose_mask.unsqueeze(2) & use_pose_mask.unsqueeze(1)
            token_interaction_mask = view_interaction_mask.repeat_interleave(hw - self.patch_start_idx, dim=1)
            token_interaction_mask = token_interaction_mask.repeat_interleave(hw - self.patch_start_idx, dim=2)
            pose_inject_mask = token_interaction_mask[:, None]

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            do_checkpoint = False
            if self.checkpoint_strategy == 'all':
                do_checkpoint = True
            elif self.checkpoint_strategy == 'global_only':
                if i % 2 != 0:
                    do_checkpoint = True

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            if self.training and do_checkpoint:
                # hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
                hidden = checkpoint(blk, hidden, xpos=pos, attn_mask=None, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos, attn_mask=None)

            if self.use_multimodal:
                if i in [1, 9, 17, 25, 33]:

                    hidden = hidden.reshape(B, N, -1, 1024)
                    if self.training and do_checkpoint:
                        poses_feat = checkpoint(self.pose_inject_blk[pose_inject_blk_idx], hidden[..., self.patch_start_idx:, :].reshape(B, N*(hw-self.patch_start_idx), -1), poses, H, W, H//14, W//14, attn_mask=pose_inject_mask, use_reentrant=False).reshape(B, N, -1, 1024)
                    else:
                        poses_feat = self.pose_inject_blk[pose_inject_blk_idx](hidden[..., self.patch_start_idx:, :].reshape(B, N*(hw-self.patch_start_idx), -1), poses, H, W, H//14, W//14, attn_mask=pose_inject_mask).reshape(B, N, -1, 1024)
                    hidden[..., self.patch_start_idx:, :] += poses_feat * use_pose_mask.view(B, N, 1, 1)
                    
                    hidden = hidden.reshape(B, N*hw, -1)
                    pose_inject_blk_idx += 1

            if i == len(self.decoder) - 2:
                temp_features = hidden.clone().reshape(B*N, hw, -1)

        concatenated = torch.cat((temp_features, hidden.reshape(B*N, hw, -1)), dim=-1)

        return concatenated, pos.reshape(B*N, hw, -1), ref_idxs
    
    
    def normalize_depth(self, depths: torch.Tensor, method: str = 'median') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes a batch of depth maps using either median or mean normalization.

        Args:
            depths (torch.Tensor): A batch of depth maps with shape [B, N, H, W].
                                Non-positive values are treated as invalid depth data.
            method (str, optional): The normalization method to use.
                                    Can be 'median' or 'mean'. Defaults to 'median'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized depth maps.
                - The normalization factors (medians or means) used for each batch element.

        Raises:
            ValueError: If the method is not 'median' or 'mean'.
        """
        # 确保输入是 torch.Tensor
        if not isinstance(depths, torch.Tensor):
            depths = torch.tensor(depths, dtype=torch.float32)

        if method not in ['median', 'mean']:
            raise ValueError(f"Invalid normalization method: '{method}'. Choose 'median' or 'mean'.")

        B, N, H, W = depths.shape
        epsilon = 1e-8

        # Create a mask for valid depth values (positive values)
        valid_depths = torch.where(depths > 0, depths, float('nan'))
        valid_depths_reshaped = valid_depths.view(B, -1)

        if method == 'median':
            # Calculate the median for each depth map in the batch
            factors, _ = torch.nanmedian(valid_depths_reshaped, dim=1)
        elif method == 'mean':
            # Calculate the mean for each depth map in the batch
            factors = torch.nanmean(valid_depths_reshaped, dim=1)
        
        # Handle cases where all values might be NaN (e.g., all depths are 0 or negative)
        # In such cases, use 1.0 as the normalization factor to prevent division by zero.
        factors = torch.nan_to_num(factors, nan=1.0)
        
        # Reshape factors for broadcasting during division
        factors_for_division = factors.view(B, 1, 1, 1)

        # Perform normalization, adding a small epsilon to prevent division by zero
        normalized_depths = depths / (factors_for_division + epsilon)

        return normalized_depths, factors.reshape(-1)
    
class CameraHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous()) 
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]

        with torch.amp.autocast(device_type='cuda', enabled=False):
            feat = feat.float()

            out_t = self.fc_t(feat)  # [B,3]
            out_r = self.fc_rot(feat)  # [B,9]

            pose = self.convert_pose_to_4x4(out_r.shape[0], out_r, out_t, feat.device)

        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r
    