# model setting
model = dict(
    type="VecSetAEV4CTF",
    input_pcd_num = 50000,
    num_latent = 2048,
    latent_dim = 512,
    use_skip = False,
    use_cpu_fps = True,
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
        enable_ca_flash=False,
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
        enable_ca_flash=False,
        shuffle_orders=False,
        cls_mode=True, #True for enc only; False for enc-dec backbone        
    ),
    decoder_dict1 = dict(
        type = "VecSetDecoderV2", #['VecSetDecoder','VecSetKLDecoder']
        depth = 6, #24,
        dim = 512,
        queries_dim = 512,
        enable_flash=True,
        attn_dtype='bf16',
        enable_ca_flash=False,
        drop_path_rate=0.3,
        output_dim = 1,
        # deterministic = True, # ['Auto' or 'KL'] ## TODO
    ),
    decoder_dict2 = dict(
        type = "VecSetDecoderV2", #['VecSetDecoder','VecSetKLDecoder']
        depth = 6, #24,
        dim = 512,
        queries_dim = 512,
        enable_flash=True,
        attn_dtype='bf16',
        enable_ca_flash=False,
        drop_path_rate=0.3,
        output_dim = 1,
        # deterministic = True, # ['Auto' or 'KL'] ## TODO
    ),
    decoder_dict3 = dict(
        type = "VecSetDecoderV2", #['VecSetDecoder','VecSetKLDecoder']
        depth = 6, #24,
        dim = 512,
        queries_dim = 512,
        enable_flash=True,
        attn_dtype='bf16',
        enable_ca_flash=False,
        drop_path_rate=0.3,
        output_dim = 1,
        # deterministic = True, # ['Auto' or 'KL'] ## TODO
    ),
    decoder_dict4 = dict(
        type = "VecSetDecoderV2", #['VecSetDecoder','VecSetKLDecoder']
        depth = 6, #24,
        dim = 512,
        queries_dim = 512,
        enable_flash=True,
        attn_dtype='bf16',
        enable_ca_flash=False,
        drop_path_rate=0.3,
        output_dim = 1,
        # deterministic = True, # ['Auto' or 'KL'] ## TODO
    ),
)


# dataset settings
batch_size = 1 # currently only support batchsize = 1
num_workers = 1
data_root = "/mnt/nvme3/z00533370/dataset/scannetpp_norm_v6/"
val_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_val.txt'
data_args = dict(
    vs_res = [16,32,64,128],
    query_sample_rate = 1.0,
)

# eval metric
occ_threshold = 0.5

# ckpt and output result
root = '/mnt/nvme3/z00533370/exp_result/'
ckpt_name = 'exp0618_VecSetAE_PTV3_NetArc_V6-2_4gpu_datav2_latent2048_scannetpp_normv6_ctf_4level_coswarmup_nolossweight_gt_pretrain/003-VecSetAEV4CTF/epoch33-global_step7000/model_ckpt.pt'
save_as_pt = True
outputs = "/mnt/nvme3/z00533370/test_result/"
prefix = "eval_mode_new_metric_gt"

# Others
# outputs = "/mnt/nvme3/z00533370/exp_result/exp0620_3D_geometry_vae_normv6_ctf_4level_debug/"
