import torch.nn as nn
from .structured_latent_vae.encoder_ori import SLatEncoderOri
from .structured_latent_vae.encoder import SLatEncoder
from .structured_latent_vae.decoder_gs import SLatGaussianDecoderOri 
from .structured_latent_vae.decoder_gs_newVox import SLatGaussianDecoder
from .structured_latent_vae.decoder_gs_newVox_sepAttr import SLatGaussianSepMLPDecoder

from ..registry import MODELS

@MODELS.register_module()
class SparseLatentVAEtrellis(nn.Module):
    def __init__(self, in_channels=1024, model_channels=768, resolution=(64,64,64),  # voxel_size=0.1, 
        latent_channels=8, num_blocks=12, num_heads=12, num_head_channels=64,
        mlp_ratio=4, attn_mode='swin', #["full", "shift_window", "shift_sequence", "shift_order", "swin"]
        window_size=8, pe_mode="ape", #["ape", "rope"]
        use_fp16=False, use_bf16=False, 
        use_checkpoint=False, qk_rms_norm=False, representation_config=None):
        super().__init__()
        # encode 默认output channel是latent_channels*2
        self.model_encoder = SLatEncoderOri(
                                         resolution=resolution,
                                         in_channels=in_channels,
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,)
        self.model_decoder = SLatGaussianDecoderOri(
                                         resolution=resolution,
                                        #  voxel_size=voxel_size,
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,
                                         representation_config=representation_config)

    def encode(self, x, sample_posterior=False):
        """
        arg:
            x : input [B, in_channels]
        return:
            latent: [B, latent_channels] 
        """
        latent = self.model_encoder(x, sample_posterior=sample_posterior)
        return latent

    def decode(self, latent):
        """
        arg:
            latent : [B, latent_channels] 
        return:
            gs: list: trellis.representations.gaussian.gaussian_model.Gaussian object
        """
        gs = self.model_decoder(latent)
        return gs

    def forward(self, x, sample_posterior=False):
        latent = self.model_encoder(x, sample_posterior=sample_posterior) # latent: latent.coords [11887, 4], latent.feats: [11887, 8], 
        gs = self.model_decoder(latent)
        return gs


@MODELS.register_module()
class SparseLatentVAEscannet(nn.Module):
    def __init__(self, in_channels=1024, model_channels=768, voxel_size=0.1, # resolution=(64,64,64), 
        latent_channels=8, num_blocks=12, num_heads=12, num_head_channels=64,
        dec_num_hidden_dims=4, enc_num_hidden_dims=1,
        mlp_ratio=4, attn_mode='full', #["full", "shift_window", "shift_sequence", "shift_order", "swin"]
        window_size=768, pe_mode="ape", #["ape", "rope"]
        use_fp16=False, use_bf16=False, voxel_mode="center", # voxel_bound=(0,0.1), 
        use_checkpoint=False, qk_rms_norm=False, representation_config=None):
        super().__init__()
        # encode 默认output channel是latent_channels*2
        self.model_encoder = SLatEncoder(
                                         in_channels=in_channels,
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,
                                         num_hidden_dims=enc_num_hidden_dims)
        self.model_decoder = SLatGaussianDecoder(
                                         # resolution=resolution,
                                        #  voxel_size=voxel_size,
                                        #  voxel_bound=voxel_bound,
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         voxel_mode=voxel_mode,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,
                                         num_hidden_dims=dec_num_hidden_dims,
                                         representation_config=representation_config)
        print("model_encoder: \n", self.model_encoder)
        self.num_gaussians = representation_config['num_gaussians']

    def encode(self, x, sample_posterior=True, return_raw=False):
        """
        arg:
            x : input [B, in_channels]
        return:
            latent: [B, latent_channels] 
        """
        if return_raw:
            latent, mean, logvar = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
            return latent, mean, logvar
        else:
            latent = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
            return latent

    def decode(self, latent, grid_origin=None, voxelGS_xyz=None, ):
        """
        arg:
            latent : [B, latent_channels] 
        return:
            gs: list: trellis.representations.gaussian.gaussian_model.Gaussian object
        """
        gs = self.model_decoder(latent, grid_origin, voxelGS_xyz)
        return gs

    def forward(self, x, grid_origin=None, voxelGS_xyz=None, sample_posterior=True, return_raw=False):
        """
        voxelGS_xyz : [N , K, 3]
        """
        if return_raw:
            latent, mean, logvar = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
        else:
            latent = self.model_encoder(x, sample_posterior=sample_posterior) # latent: latent.coords [11887, 4], latent.feats: [11887, 8], 
        if voxelGS_xyz is not None:
            voxelGS_xyz = voxelGS_xyz.contiguous().view(x.feats.shape[0], self.num_gaussians*3)
        gs = self.model_decoder(latent, grid_origin, voxelGS_xyz)
        if return_raw:
            return gs, mean, logvar
        else:
            return gs


@MODELS.register_module()
class SparseLatentCrossAttnVAEscannet(nn.Module):
    def __init__(self, in_channels=1024, model_channels=768, # resolution=(64,64,64), voxel_size=0.1,
        latent_channels=8, num_blocks=12, num_heads=12, num_head_channels=64,
        dec_num_hidden_dims=4, enc_num_hidden_dims=1, num_latent=4096,
        mlp_ratio=4, attn_mode='full', #["full", "shift_window", "shift_sequence", "shift_order", "swin"]
        window_size=768, pe_mode="ape", #["ape", "rope"]
        use_fp16=False, use_bf16=False, # voxel_bound=(0,0.1), 
        use_checkpoint=False, qk_rms_norm=False, representation_config=None):
        super().__init__()
        # encode 默认output channel是latent_channels*2
        self.model_encoder = SLatEncoder(
                                         in_channels=in_channels,
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         num_latent=num_latent,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,
                                         num_hidden_dims=enc_num_hidden_dims)
        self.model_decoder = SLatGaussianDecoder(
                                         # resolution=resolution,
                                        #  voxel_size=voxel_size,
                                        #  voxel_bound=voxel_bound,
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,
                                         num_hidden_dims=dec_num_hidden_dims,
                                         representation_config=representation_config)
        print("model_encoder: \n", self.model_encoder)
        self.num_gaussians = representation_config['num_gaussians']

    def encode(self, x, sample_posterior=True, return_raw=False):
        """
        arg:
            x : input [B, in_channels]
        return:
            latent: [B, latent_channels] 
        """
        if return_raw:
            latent, mean, logvar = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
            return latent, mean, logvar
        else:
            latent = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
            return latent

    def decode(self, latent, query, grid_origin=None, voxelGS_xyz=None, ):
        """
        arg:
            latent : [B, latent_channels] 
        return:
            gs: list: trellis.representations.gaussian.gaussian_model.Gaussian object
        """
        gs = self.model_decoder(latent, query, grid_origin, voxelGS_xyz)
        return gs

    def forward(self, x, grid_origin=None, voxelGS_xyz=None, sample_posterior=True, return_raw=False):
        """
        voxelGS_xyz : [N , K, 3]
        """
        if return_raw:
            latent, mean, logvar = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
        else:
            latent = self.model_encoder(x, sample_posterior=sample_posterior) # latent: latent.coords [11887, 4], latent.feats: [11887, 8], 
        if voxelGS_xyz is not None:
            voxelGS_xyz = voxelGS_xyz.contiguous().view(x.feats.shape[0], self.num_gaussians*3)
        gs = self.model_decoder(latent, grid_origin, voxelGS_xyz)
        if return_raw:
            return gs, mean, logvar
        else:
            return gs


@MODELS.register_module()
class SparseLatentVAESepMLPscannet(nn.Module):
    def __init__(self, in_channels=1024, model_channels=768, # resolution=(64,64,64), voxel_size=0.1,
        latent_channels=8, num_blocks=12, num_heads=12, num_head_channels=64,
        dec_num_hidden_dims=4, enc_num_hidden_dims=1,        
        mlp_ratio=4, attn_mode='swin', #["full", "shift_window", "shift_sequence", "shift_order", "swin"]
        window_size=768, pe_mode="ape", #["ape", "rope"]
        use_fp16=False, use_bf16=False, # voxel_bound=(0,0.1), 
        use_checkpoint=False, qk_rms_norm=False, representation_config=None):
        super().__init__()
        # encode 默认output channel是latent_channels*2
        self.model_encoder = SLatEncoder(
                                         in_channels=in_channels,
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,
                                         num_hidden_dims=enc_num_hidden_dims
                                         )
        self.model_decoder = SLatGaussianSepMLPDecoder(
                                         # resolution=resolution,
                                        #  voxel_size=voxel_size,
                                        #  voxel_bound=voxel_bound, 
                                         model_channels=model_channels,
                                         latent_channels=latent_channels,
                                         num_blocks=num_blocks,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         mlp_ratio=mlp_ratio,
                                         attn_mode=attn_mode,
                                         window_size=window_size,
                                         pe_mode=pe_mode,
                                         use_fp16=use_fp16,
                                         use_bf16=use_bf16,
                                         use_checkpoint=use_checkpoint,
                                         qk_rms_norm=qk_rms_norm,
                                         num_hidden_dims=dec_num_hidden_dims,
                                         representation_config=representation_config
                                         )
        self.num_gaussians = representation_config['num_gaussians']

    def encode(self, x, sample_posterior=True, return_raw=False):
        """
        arg:
            x : input [B, in_channels]
        return:
            latent: [B, latent_channels] 
        """
        if return_raw:
            latent, mean, logvar = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
            return latent, mean, logvar
        else:
            latent = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
            return latent

    def decode(self, latent, grid_origin=None, voxelGS_xyz=None):
        """
        arg:
            latent : [B, latent_channels] 
        return:
            gs: list: trellis.representations.gaussian.gaussian_model.Gaussian object
        """
        gs = self.model_decoder(latent, grid_origin, voxelGS_xyz)
        return gs

    def forward(self, x, grid_origin=None, voxelGS_xyz=None, sample_posterior=True, return_raw=False):
        """
        voxelGS_xyz : [N , K, 3]
        """
        if return_raw:
            latent, mean, logvar = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)
        else:
            latent = self.model_encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)# latent: latent.coords [11887, 4], latent.feats: [11887, 8], 
        if voxelGS_xyz is not None:
            voxelGS_xyz = voxelGS_xyz.contiguous().view(x.feats.shape[0], self.num_gaussians*3)

        gs = self.model_decoder(latent, grid_origin, voxelGS_xyz)
        if return_raw:
            return gs, mean, logvar
        else:
            return gs

    # def update_learning_rate(self, optimizer, iteration):
    #     for param_group in optimizer.param_groups:
    #         if param_group["name"] == "mlp_xyz":
    #             lr = optimization.xyz_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_opa":
    #             lr = optimization.opa_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_sca":
    #             lr = optimization.sca_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_rot":
    #             lr = optimization.rot_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_fdc":
    #             lr = optimization.fdc_scheduler_args(iteration)
    #             param_group['lr'] = lr

