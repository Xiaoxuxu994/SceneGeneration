# acceleration & misc setting
dtype = "fp32"  # fp16, bf16, fp32, 
grad_checkpoint = True
plugin = "zero2"
sp_size = 1
seed = 42
init_with_xavier = True

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
num_workers = 8
data_root = "/pytorch210_910b_share/datasets/scannetpp_norm_v3_02_vs16_256"
train_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train2.txt'
val_data_list_name = ['datacache/scannetpp_trainval/nvs_sem_val.txt']
vs_res = [256, 512, 'freequery']
vs_res_gd =256
truncatedist = 5 * 2.0 / 256
train_data_args = dict(
    datatype = 'npz',
    vs_res = vs_res, #  gen data[key](key in vs_res):pc_query_{vs}, pc_query_sdf_{vs}
    freequery_root = '"/home/ma-user/modelarts/inputs/freequery_root_1/', # 补充的freequery数据
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
        'max_number': 66667,
        'vs_list': vs_res, # sample data[key](key in vs_list)
        'near_thresh': [3 * 2.0 / 256, None, None],
        'near_ratio': [0.8, None, None],
        'sample_type': ['nearfar','uniform','uniform'],
    },
    'MergeDecodeQuery':{ # merge data(key in vs_list) into pc_query_merge, pc_query_sdf_merge for train&val
        'enabled': True,
        'vs_list': vs_res,
    },
    'ToTensor': {'enabled': True}
    }
)

vs_res_val = [256]
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
        'vs_list': vs_res_val, # decoder voxel size, TODO:support coarse-to-fine
        'near_thresh': [3 * 2.0 / vs_res_val[0]],
        'near_ratio': [0.8],
        'sample_type': ['nearfar']
    },
    'ToTensor': {'enabled': True}
    }
)

# scheduler settings
epochs = 1500
gt_warmup_epochs = 30

# 监控各层输出开关
monitor = False
monitor_every = 1
log_every = 500
ckpt_every = 500
eval_every = 500
load = None
# resume = "output/test1/model/"
resume = None

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

# softclip
softclip = True
softcliptype = 'double_leaky_relu'
softclipalpha = 1e-3
epoch_ratio = 1.0 # 0.0=always hardclip, 1.0=always softclip

## KL loss
loss_weight_KL = 1e-3

## sdf loss
enable_sdf_loss = True
sdf_loss_type = 'mse' # ['l1','mse']
loss_weight_SDF = [10000.0]

#eval metric loss
occ_threshold = 0.5 # eval metric loss after sigmoid
sdf_threshold = 2.0 / vs_res_val[-1]
use_cpu_metric = False # use simplified torchmetric to speed up training

# 指定输出路径, 如果如下则模型ckpt位于output/test2/model/global_stepxxx, 训练log位于output/test2/tensorboard
outputs = "output/test2"
exp_name = ""
