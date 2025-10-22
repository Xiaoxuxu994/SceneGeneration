# acceleration & misc setting
dtype = "fp32"  # fp16, bf16, fp32, 
grad_checkpoint = True
plugin = "zero2"
sp_size = 1
seed = 42
init_with_xavier = False

# model setting
dim = 1024
model = dict(
    type="VecSetAEV6SINGLEKL",
    num_latent = 4096,
    latent_dim = dim,
    kl_dim = 64,
    chunk_size = 200000, # for [129*129*129]
    use_skip = True,
    use_cpu_fps = False,
    use_kernel_fps = True,
    encoder_dict = dict(type = "PE",),
    query_encoder_dict = dict(type = "PE",),
    # transformer_dict = None,
    transformer_dict = dict(
        type = 'TransformerEncoderBlock',
        depth = 8,
        dim = dim,
        heads = 8,
        dim_head = 128,
        enable_flash = True,
        attn_dtype = 'bf16',
        drop_path_rate = 0.3,
    ),
    decoder_dict4 = dict(
        type = "VecSetDecoderV1", 
        depth = 16,
        dim = dim,
        queries_dim = dim,
        heads = 8,
        dim_head = 128,
        enable_flash=True,
        attn_dtype='bf16',
        enable_ca_flash=False,
        drop_path_rate=0.3,
        output_dim = 1,
    ), ## vs_res=128
)


# dataset settings
batch_size = 1 # currently only support batchsize = 1
num_workers = 8
# data_root = "/mnt/nvme1/share/datasets/scannetpp_norm_v3_02_vs16_256_oct512_3x/"
data_root = "/efs-mi-east4-2/datasets/3DGeoVAE_data/"
train_data_list_name = 'datacache/mix_trainval/scannetpp_arkitscene_train_v1.txt'
val_data_list_name = ['datacache/mix_trainval/scannetpp_val.txt','datacache/mix_trainval/arkitscene_val.txt']
# val_data_list_name = 'datacache/arkitscene_trainval/arkitscene_val.txt'
val_names = ['scannet','arkitscene']
vs_res = [256,512,'fq']
vs_res_trainkey = ['merge']
vs_res_valkey = [256,512,'fq','merge']
vs_res_gd = 256
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
        'max_number': 100000,
        'vs_list': vs_res, # decoder voxel size, TODO:support coarse-to-fine
        'near_thresh': [3 * 2.0 / 256, None, 3 * 2.0 / 256],
        'near_ratio': [0.8, None, 0.8],
        'sample_type': ['nearfar','uniform','nearfar'],
    },
    'MergeDecodeQuery':{ # merge data(key in vs_list) into pc_query_merge, pc_query_sdf_merge for train&val
        'enabled': True,
        'vs_list': vs_res,
    },
    'ToTensor': {'enabled': True},
    'Collect':{
        'enabled':True,
        'keys':['coord','normal','scene_id',
        'pc_query_256','pc_query_sdf_256',
        'pc_query_512','pc_query_sdf_512',
        'pc_query_fq','pc_query_sdf_fq',
        'pc_query_merge','pc_query_sdf_merge']}
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
        'max_number': 100000,
        'vs_list': vs_res, # decoder voxel size, TODO:support coarse-to-fine
        'near_thresh': [3 * 2.0 / 256, None, 3 * 2.0 / 256],
        'near_ratio': [0.8, None, 0.8],
        'sample_type': ['nearfar','uniform','nearfar'],
    },
    'MergeDecodeQuery':{ # merge data(key in vs_list) into pc_query_merge, pc_query_sdf_merge for train&val
        'enabled': True,
        'vs_list': vs_res,
    },
    'ToTensor': {'enabled': True},
    'Collect':{
        'enabled':True,
        'keys':['coord','normal','scene_id',
        'pc_query_256','pc_query_sdf_256',
        'pc_query_512','pc_query_sdf_512',
        'pc_query_fq','pc_query_sdf_fq',
        'pc_query_merge','pc_query_sdf_merge']}
    }
)

# scheduler settings
epochs = 1500
gt_warmup_epochs = 30

# 监控各层输出开关
monitor = False
monitor_every = 1
log_every = 1 #500
ckpt_every = 1 #500
eval_every = 1 #500

resume = None
load = None

lr = 1e-4 
lr_scheduler = 'cosine'
min_lr_ratio=0.05
warmup_ratio=0.1
grad_clip = 1.0
weight_decay = 0 #0.1


# soft clip
softclip = True
softcliptype = 'double_leaky_relu'
softclipalpha = 0.001
epoch_ratio = 1.0

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
loss_weight_SDF = [10000.0]  * len(vs_res_valkey) ## 指标在train的时候只对应第一维度，eval的时候对应多个

#eval metric loss
occ_threshold = 0.5 # eval metric loss after sigmoid
sdf_threshold = [2.0 / 256] * len(vs_res_valkey)
truncatedist = [3.0 * 2.0 / 256] * len(vs_res_valkey)
truncatedist_train = [5.0 * 2.0 / 256] * len(vs_res_valkey)

# Others
outputs = "/mnt/nvme3/z00533370/exp_result/exp1022_mix_multidata_debug/"
exp_name = ""
