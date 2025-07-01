# acceleration & misc setting
dtype = "fp32"  # fp16, bf16, fp32, 
grad_checkpoint = True
plugin = "zero2"
sp_size = 1
seed = 42

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
    use_cpu_fps=True
)

# dataloader
batch_size = 1 # currently only support batchsize = 1
num_workers = 1
query_sample_rate = 0.1
max_query_number = 100000
grid_size = 0.05

# dataset settings
data_root = '../../data/scannetpp_pcd'
train_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train2.txt'
val_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_val.txt'

# scheduler settings
epochs = 1
log_every = 500
ckpt_every = 500
resume = None
load = None

lr = 1e-4 
cos_total_iter = 5000 
cos_eta_min = 5e-7
grad_clip = 1.0
weight_decay = 0


# loss & eval
# - occ geometry loss
enable_occ_loss = True
loss_weight_occ = 50
# - eval metric
enable_occ_eval = False
eval_metric_list = ['precision','recall','f1','iou']
occ_threshold = 0.0

# Others
outputs = "./test/expr-match"



