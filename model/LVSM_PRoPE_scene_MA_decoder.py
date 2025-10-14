# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils
from .loss import LossComputer
from .mapanything_encoder import MapAnythingEncoder
from .prope_transformer import TransformerEncoder, PropeDotProductAttention
 

class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize target pose tokenizer
        self.target_pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.target_pose_tokenizer.patch_size,
                pw=self.config.model.target_pose_tokenizer.patch_size,
            ),
            nn.Linear(
                self.config.model.target_pose_tokenizer.in_channels * (self.config.model.target_pose_tokenizer.patch_size**2),
                self.config.model.transformer.d,
                bias=False,
            ),
        )
        
        # Initialize (frozen) MapAnything
        self.mapanything_encoder = MapAnythingEncoder.from_pretrained("facebook/map-anything")
        for param in self.mapanything_encoder.parameters():
            param.requires_grad = False

        # Initialize feature fusion module
        self.feature_fusion = nn.Conv1d(
            in_channels=self.mapanything_encoder.encoder.enc_embed_dim + self.config.model.transformer.d*3,  # TODO: remove hard-coded
            out_channels=self.config.model.transformer.d, 
            kernel_size=1
        )
        
        # Initialize loss computer
        self.loss_computer = LossComputer(config)
        
        # PRoPE attention
        self.attention = PropeDotProductAttention(
            head_dim=self.config.model.transformer.d_head,
            patches_x=518 // self.config.model.target_pose_tokenizer.patch_size,
            patches_y=294 // self.config.model.target_pose_tokenizer.patch_size,
            image_width=518,
            image_height=294,
        )  # TODO: remove hard-coded
        
        # Initialize transformer decoder
        self.encoder = TransformerEncoder(
            d_model=self.config.model.transformer.d,
            nhead=self.config.model.transformer.d // self.config.model.transformer.d_head,
            num_layers=self.config.model.transformer.decoder_n_layer,
            qk_norm=self.config.model.transformer.get("use_qk_norm", False),
            dim_feedforward=4*self.config.model.transformer.d
        )
        
        # Initialize output layer
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        
        self.init_weights()

    
    def init_weights(self):
        for idx, layer in enumerate(self.encoder.layers):
            layer.apply(self.init_layer_weights(idx))


    def init_layer_weights(self, idx):
        # LVMS Paper A.1:
        # "We initialize the model weights with a normal distribution of zero-mean
        # and standard deviation of 0.02/(2 * (idx+ 1)) ** 0.5, where idx means
        # transform layer index."
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02 / (2 * (idx + 1)) ** 0.5)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        return _init_weights


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()
            

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
    
    
    def forward(self, data_batch, has_target_image=True):        
        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, 
                                          compute_rays=True, compute_rays_camera_frame=True, pose_to_quat=True)
        
        # Encode input images and poses with MapAnything
        aggregated_tokens_list = self.mapanything_encoder.forward_feats(
            input.image,
            input.ray_d_camera_frame,
            input.camera_pose_quats,
            input.camera_pose_trans,
            input.is_metric_scale
        )
        _, embed_dim, patch_h, patch_w = aggregated_tokens_list[0].shape
        n_patches = patch_h * patch_w
        
        # # Fuse MapAnything tokens
        # latent_tokens = self.feature_fusion(aggregated_tokens_list)
        # raise ValueError(latent_tokens.shape)
        
        # Fuse MapAnything tokens (simple Conv feature fusion)
        latent_tokens = torch.cat(aggregated_tokens_list, dim=1)  # [b*v_input, embed_dim*4, patch_h, patch_w]
        latent_tokens = latent_tokens.flatten(2)  # [b*v_input, embed_dim*4, n_patches] for Conv1d
        latent_tokens = self.feature_fusion(latent_tokens)  # [b*v_input, d, n_patches]
        b = self.config.training.batch_size_per_gpu
        v_input = self.config.training.num_input_views
        latent_tokens = latent_tokens.reshape(b, v_input, -1, n_patches).permute(0, 1, 3, 2).contiguous()  # [b, v_input, n_patches, d]
        b, v_input, n_patches, d = latent_tokens.shape
        n_latent_vectors = v_input*n_patches
        latent_tokens = latent_tokens.reshape(b, n_latent_vectors, d).contiguous()  # treat n_patches and v_input as sequence length (n_latent_vectors)
        
        # Get target pose tokens (LVSM way)
        mapanything_patch_size = 14
        if self.config.model.target_pose_tokenizer.patch_size != mapanything_patch_size:
            # Reshape target image (and recompute rays) to use LVSM pre-trained decoder
            b, v_target, c, h, w = target.image.shape
            patch_size = self.config.model.target_pose_tokenizer.patch_size
            resize_h = self.config.model.target_pose_tokenizer.image_size
            resize_w = int(resize_h / h * w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            target.image = target.image.contiguous().view(b * v_target, c, h, w)
            target.image = F.interpolate(
                target.image,
                size=(resize_h, resize_w),
                mode="bilinear",
                align_corners=False
            )
            target.image = target.image.contiguous().view(b, v_target, c, resize_h, resize_w)
            target.image_h_w = resize_h, resize_w
            target.ray_o, target.ray_d = self.process_data.compute_rays(
                c2w=target.c2w, fxfycxcy=target.fxfycxcy, h=resize_h, w=resize_w,
            )
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)  # [b, v_target, c, resize_h, resize_w]
        b, v_target, c, resize_h, resize_w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)  # [b*v_target, n_patches_target, d]
        _, n_patches_target, d = target_pose_tokens.shape
        
        # --- Prepare data for geomtry-aware self-attention ---
        ref_c2ws = input.c2w
        ref_c2ws = repeat(ref_c2ws, "b v_input x y -> (b v_target) v_input x y", v_target=v_target)
        fx, fy, cx, cy = input.fxfycxcy.unbind(-1)
        ref_Ks = torch.zeros((*input.fxfycxcy.shape[:-1], 3, 3), device=input.fxfycxcy.device, dtype=input.fxfycxcy.dtype)
        ref_Ks[..., 0, 0], ref_Ks[..., 1, 1], ref_Ks[..., 0, 2], ref_Ks[..., 1, 2], ref_Ks[..., 2, 2] = fx, fy, cx, cy, 1
        ref_Ks = repeat(ref_Ks, "b v_input x y -> (b v_target) v_input x y", v_target=v_target)
        tar_c2ws = target.c2w
        tar_c2ws = rearrange(tar_c2ws, "b v_target x y -> (b v_target) 1 x y", v_target=v_target)
        fx, fy, cx, cy = target.fxfycxcy.unbind(-1)
        tar_Ks = torch.zeros((*target.fxfycxcy.shape[:-1], 3, 3), device=target.fxfycxcy.device, dtype=target.fxfycxcy.dtype)
        tar_Ks[..., 0, 0], tar_Ks[..., 1, 1], tar_Ks[..., 0, 2], tar_Ks[..., 1, 2], tar_Ks[..., 2, 2] = fx, fy, cx, cy, 1
        tar_Ks = rearrange(tar_Ks, "b v_target x y -> (b v_target) 1 x y")
        c2ws = torch.cat([ref_c2ws, tar_c2ws], dim=1)  # [B, N, 4, 4] per camera
        Ks = torch.cat([ref_Ks, tar_Ks], dim=1)  # [B, N, 3, 3] per camera
        viewmats = torch.inverse(c2ws)

        def sdpa_fn(q, k, v, **kwargs):            
            return self.attention(q, k, v, viewmats=viewmats, Ks=Ks, **kwargs)

        # Repeat latent tokens along the batch dimension to match target tokens batch dimension
        repeated_latent_tokens = repeat(
                                latent_tokens,
                                'b nl d -> (b v_target) nl d', 
                                v_target=v_target)  # [b*v_target, n_latent_vectors, d]
        
        # run attentions
        xq = torch.cat([repeated_latent_tokens, target_pose_tokens], dim=1) 
        try:
            xq = self.encoder(xq, sdpa_fn=sdpa_fn)
        except:
            raise ValueError(input.image.shape, target.image.shape)
        
        # discard latent tokens
        q = xq[:, -n_patches_target:, :]

        # output layer
        rendered_images = self.image_token_decoder(q)
        height, width = target.image_h_w
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )
        if has_target_image:
            # Denormalize predicted and target views
            rendered_images = data_utils.denormalize(rendered_images, norm_type='identity')
            target.image = data_utils.denormalize(target.image, norm_type='identity')
            input.image = data_utils.denormalize(input.image, norm_type='identity')
            loss_metrics = self.loss_computer(
                rendered_images,
                target.image
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images
        )
        
        return result


    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        images = input.images
        
        # Encode images with VGGT
        aggregated_tokens_list, _ = self.vggt_encoder(images)
        
        latent_tokens = None  # TODO: feature fusing from VGGT DPT      

        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4
        video_rendering_list = []
        
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)
            
            # Get current chunk of target pose tokens
            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_repeated_latent_tokens = repeat(
                latent_tokens,
                'b nl d -> (b v_chunk) nl d', 
                v_chunk=cur_view_chunk_size
                )

            decoder_input_tokens = torch.cat((cur_target_pose_tokens, cur_repeated_latent_tokens), dim=1)
            decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

            transformer_output_tokens = self.pass_layers(
                self.transformer_decoder, 
                decoder_input_tokens, 
                gradient_checkpoint=False
            )

            target_image_tokens, _ = transformer_output_tokens.split(
                [n_patches, self.config.model.transformer.n_latent_vectors], dim=1
            )

            # Decode to images
            height, width = target.image_h_w
            patch_size = self.config.model.target_pose_tokenizer.patch_size
            
            video_rendering = self.image_token_decoder(target_image_tokens)
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)

        # Combine all chunks
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering

        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=False)
        return 0
