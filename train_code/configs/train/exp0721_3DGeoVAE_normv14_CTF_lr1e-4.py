# acceleration & misc setting
dtype = "fp32"  # fp16, bf16, fp32, 
grad_checkpoint = True
plugin = "zero2"
sp_size = 1
seed = 42

# model setting
model = dict(
    type="VecSetAEV4CTFKL",
    num_latent = 2048,
    latent_dim = 512,
    kl_dim = 64,
    chunk_size = 200000, # for [129*129*129]
    use_skip = True,
    use_cpu_fps = True,
    encoder_dict = dict(type = "PE",),
    query_encoder_dict = dict(type = "PE",),
    # transformer_dict = None,
    transformer_dict = dict(
        type = 'TransformerEncoderBlock',
        depth = 4,
        dim = 512,
        heads = 8,
        dim_head = 64,
        enable_flash = True,
        attn_dtype = 'bf16',
        drop_path_rate = 0.3,
    ),
    decoder_dicts = [
        dict(
            type = "VecSetDecoderV2", 
            depth = 2,
            dim = 512,
            queries_dim = 512,
            enable_flash=True,
            attn_dtype='bf16',
            enable_ca_flash=False,
            drop_path_rate=0.3,
            output_dim = 1,), ## vs_res=16
        dict(
            type = "VecSetDecoderV2",
            depth = 4,
            dim = 512,
            queries_dim = 512,
            enable_flash=True,
            attn_dtype='bf16',
            enable_ca_flash=False,
            drop_path_rate=0.3,
            output_dim = 1,), ## vs_res=32
        dict(
            type = "VecSetDecoderV2", 
            depth = 6,
            dim = 512,
            queries_dim = 512,
            enable_flash=True,
            attn_dtype='bf16',
            enable_ca_flash=False,
            drop_path_rate=0.3,
            output_dim = 1,), ## vs_res=64
        dict(
            type = "VecSetDecoderV2", 
            depth = 8,
            dim = 512,
            queries_dim = 512,
            enable_flash=True,
            attn_dtype='bf16',
            enable_ca_flash=False,
            drop_path_rate=0.3,
            output_dim = 1,), ## vs_res=128
    ]
)


# dataset settings
batch_size = 1 # currently only support batchsize = 1
num_workers = 1
data_root = "/mnt/nvme3/z00533370/dataset/scannetpp_norm_v14/"
train_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train2.txt'
val_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_val.txt'
vs_res = [16,32,64,128]
train_data_args = dict(
    vs_res = vs_res,
    transform = {
    'GridSample': {
        'enabled': True,
        'grid_size': 2.0 / 128, # encoder voxelsize
        'hash_type': 'fnv',
        'mode': 'train', # train for both train and test
        'keys': ('coord',),
        'return_grid_coord': False
    },
    'ShufflePoint': {
        'enabled': False,
        'vs_list': vs_res}, # shuffle both input point cloud and multi-level query
    'ToTensor': {'enabled': True}
    }
)

val_data_args = dict(
    vs_res = vs_res,
    transform = {
    'GridSample': {
        'enabled': True,
        'grid_size': 2.0 / 128, # encoder voxelsize
        'hash_type': 'fnv',
        'mode': 'train', # train for both train and test
        'keys': ('coord',),
        'return_grid_coord': False
    },
    'ToTensor': {'enabled': True}
    }
)

# scheduler settings
epochs = 500 #10000
gt_warmup_epochs = 30
log_every = 500
ckpt_every = 500
resume = None
load = None

lr = 1e-4 
min_lr_ratio=0.05
warmup_ratio=0.1
grad_clip = 1.0
weight_decay = 0 #0.1


# loss & eval
# - occ geometry loss
BEC_pos_weight = False # bec pos/neg sample weight
loss_weight_occ = [1.0,1.0,1.0,1.0]
loss_weight_KL = 1e-3
occ_threshold = 0.5 # eval metric loss after sigmoid

# Others
outputs = "/mnt/nvme3/z00533370/exp_result/exp0721_3DGeoVAE_normv14_CTF_lr1e-4/"
