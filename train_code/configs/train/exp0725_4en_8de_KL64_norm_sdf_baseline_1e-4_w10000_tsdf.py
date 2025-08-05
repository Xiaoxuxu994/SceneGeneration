# acceleration & misc setting
dtype = "fp32"  # fp16, bf16, fp32, 
grad_checkpoint = True
plugin = "zero2"
sp_size = 1
seed = 42

# model setting
model = dict(
    type="VecSetAEV4SINGLEKL",
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
    decoder_dict4 = dict(
        type = "VecSetDecoderV2", 
        depth = 8,
        dim = 512,
        queries_dim = 512,
        enable_flash=True,
        attn_dtype='bf16',
        enable_ca_flash=False,
        drop_path_rate=0.3,
        output_dim = 1,
    ), ## vs_res=128
)


# dataset settings
batch_size = 1 # currently only support batchsize = 1
num_workers = 1
data_root = "/mnt/nvme1/share/datasets/scannetpp_norm_v3_02_vs16_256/"
train_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train2.txt'
val_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_val.txt'
vs_res = [128]
vs_res_gd = 128
truncatedist = 5 * 2.0 / vs_res[-1]
train_data_args = dict(
    datatype = 'npz',
    vs_res = vs_res,
    transform = {
    'GridSample': {
        'enabled': True,
        'grid_size': 2.0 / vs_res_gd, # encoder voxelsize
        'hash_type': 'fnv',
        'mode': 'train', # train for both train and test
        'keys': ('coord',),
        'return_grid_coord': False
    },
    'DecodeQuerySampleSDF': {
        'enabled': True,
        'sample_rate': 0.1,
        'max_number': 200000,
        'vs_list': vs_res, # decoder voxel size, TODO:support coarse-to-fine
        'near_thresh': 3 * 2.0 / vs_res[-1],
        'near_ratio': 0.8,
    },
    'ToTensor': {'enabled': True}
    }
)

val_data_args = dict(
    datatype = 'npz',
    vs_res = vs_res,
    transform = {
    'GridSample': {
        'enabled': True,
        'grid_size': 2.0 / vs_res_gd, # encoder voxelsize
        'hash_type': 'fnv',
        'mode': 'train', # train for both train and test
        'keys': ('coord',),
        'return_grid_coord': False
    },
    'DecodeQuerySampleSDF': {
        'enabled': True,
        'sample_rate': 0.1,
        'max_number': 200000,
        'vs_list': vs_res, # decoder voxel size, TODO:support coarse-to-fine
        'near_thresh': 3 * 2.0 / vs_res[-1],
        'near_ratio': 0.8,
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
enable_occ_loss = False 
BEC_pos_weight = False # bec pos/neg sample weight
loss_weight_occ = [1.0]

## KL loss
loss_weight_KL = 1e-3

## sdf loss
enable_sdf_loss = True
sdf_loss_type = 'mse' # ['l1','mse']
loss_weight_SDF = [10000.0]

#eval metric loss
occ_threshold = 0.5 # eval metric loss after sigmoid
sdf_threshold = 2.0 / vs_res[-1]

# Others
outputs = "/mnt/nvme3/z00533370/exp_result/exp0725_4en_8de_KL64_norm_sdf_baseline_1e-4_w10000_tsdf/"
