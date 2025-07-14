# model setting
model = dict(
    type="VecSetAEV4",
    input_pcd_num = 50000,
    num_latent = 2048,
    latent_dim = 512,
    kl_latent_dim = 64,
    enable_kl = False,
    enable_ca_flash = False,
    encoder_dict = dict(
        type = "PointTransformerV3_2",
        in_channels = 3, # ['coord]
        stride=(),
        enc_depths=(1,),
        enc_channels=(512,),
        enc_num_head=(32,),
        enc_patch_size=(1024,),
        enable_cpe_block=False,
        enable_flash=True,
        attn_dtype='bf16',
        drop_path=0.3,
        shuffle_orders=True,
        cls_mode=True, #True for enc only; False for enc-dec backbone
    ),
    query_encoder_dict = dict(
        type = "PointTransformerV3_2",
        in_channels = 3, # ['coord]
        stride=(),
        enc_depths=(1,),
        enc_channels=(512,),
        enc_num_head=(32,),
        enc_patch_size=(1024,),
        enable_cpe_block=False,
        enable_flash=True,
        attn_dtype='bf16',
        shuffle_orders=False,
        cls_mode=True, #True for enc only; False for enc-dec backbone        
    ),
    decoder_dict = dict(
        type = "VecSetDecoder", #['VecSetDecoder','VecSetKLDecoder']
        depth = 8, #24,
        dim = 512,
        queries_dim = 512,
        enable_flash = True,
        enable_ca_flash = False,
        attn_dtype='bf16',
        drop_path_rate=0.3,
        output_dim = 1,
        # deterministic = True, # ['Auto' or 'KL'] ## TODO
    ),
    use_cpu_fps=True
)

grid_size = 0.05
# ckpt and output result
root = 'exp0417_3D_geometry_vae_releaseV1_ddp/008-VecSetAEV4KL/epoch499-global_step103999'
ckpt_name = 'ckpt499.pth'
occ_threshold = 0.5
verbose = False

##  dit decode args
# input
latent_file = '../../TRELLIS-main/outputs/sl_flow_bbox_zero_dit_XL_2048x64_fp16_scannetpp_config1_inference_general/samples/inference/sample_inference.pth'
cond_file = '../../TRELLIS-main/outputs/sl_flow_bbox_zero_dit_XL_2048x64_fp16_scannetpp_config1_inference_general/samples/inference/cond_inference.pth'
# output
occ_field_file = '../../TRELLIS-main/outputs/sl_flow_bbox_zero_dit_XL_2048x64_fp16_scannetpp_config1_inference_general/samples/inference/occ_field.pth'
# output probability before sigmoid
save_prob = False
occ_prob_file = '../../TRELLIS-main/outputs/sl_flow_bbox_zero_dit_XL_2048x64_fp16_scannetpp_config1_inference_general/samples/inference/occ_probs_field.pth'
# output each latent with a single file
save_split = False
occ_field_prefix = '../../TRELLIS-main/outputs/sl_flow_bbox_zero_dit_XL_2048x64_fp16_scannetpp_config1_inference_general/samples/inference/occ_field_'
occ_prob_prefix = '../../TRELLIS-main/outputs/sl_flow_bbox_zero_dit_XL_2048x64_fp16_scannetpp_config1_inference_general/samples/inference/occ_probs_'

## vae decode args
latent_path = "/home/zhaotiao/data3/scannetpp/latent"
output_path = "/home/zhaotiao/data3/scannetpp/decode_all"
bbox_file = "/home/zhaotiao/data/bbox.csv"