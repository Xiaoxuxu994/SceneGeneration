# model setting
model = dict(
    type="VecSetAEV4",
    input_pcd_num = 50000,
    num_latent = 4096,
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

# dataloader
batch_size = 1 # currently only support batchsize = 1
num_workers = 1

# dataset
val_dataset = dict(
    type = 'DatasetForScanNetV2',
    data_root = '/mnt/data1/z00912387/scannetpp_pcd/val',
    data_list_name = 'datacache/scannetpp_trainval/nvs_sem_val.txt',
    query_sample_rate = 1.0,
    max_query_number = 100000,
    Gridsample_mode = 'train',
    grid_size = 0.05,
)

# ckpt and output result
root = './test/expr/000-VecSetAEV4/epoch499-global_step103999'
ckpt_name = 'ckpt499.pth'
save_as_pt = True
outputs = "./test/expr/000-VecSetAEV4-val/"
prefix = ""

occ_threshold = 0.5