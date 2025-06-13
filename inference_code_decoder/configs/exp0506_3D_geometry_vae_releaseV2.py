## exp0506_VecSetAE_PTV3_NetArc_V6-2_4gpu_voxel0p05_decoder8_datav2_latent2048_updataset3.py
# model setting
model = dict(
    type="VecSetAEV4",
    input_pcd_num = 50000,
    num_latent = 2048,
    latent_dim = 512,
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
    decoder_dict = dict(
        type = "VecSetDecoder", #['VecSetDecoder','VecSetKLDecoder']
        depth = 8, #24,
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

# dataloader
batch_size = 1 # currently only support batchsize = 1
num_workers = 1
# dataset settings
data_root = '/mnt/nvme3/z00533370/dataset/scannetpp_trainval_v6/'
val_data_list_name = 'datacache/nvs_sem_val.txt'
transform = {
    'GridSample': {
        'enabled': True,
        'grid_size': 0.05, # encoder voxelsize
        'hash_type': 'fnv',
        'mode': 'train', # train for both train and test
        'keys': ('coord', 'color', 'normal', 'segment'),
        'return_grid_coord': True
    },
    'DecodeQuerySample': {
        'enabled': True,
        'sample_rate': 0.1,
        'max_number': 100000,
        'prefix': '0.05', # decoder voxel size, TODO:support coarse-to-fine
        'mode':'train', # use test for validation
    },
    'ToTensor': {'enabled': True},
    'Collect': {
        'enabled': True,
        'keys': ("coord", "grid_coord", "pc_query", "pc_query_label", "scene_id"),
        'feat_keys': ("coord",)
    }
}

# eval
occ_threshold = 0.0

# ckpt and output result
root = '../ckpt/'
ckpt_name = 'exp0506_VecSetAE_PTV3_NetArc_V6-2_4gpu_voxel0p05_decoder8_datav2_latent2048_updataset3/epoch495-global_step103000'
outputs = "./test_result_geometryvae/"
prefix = "eval_mode"
