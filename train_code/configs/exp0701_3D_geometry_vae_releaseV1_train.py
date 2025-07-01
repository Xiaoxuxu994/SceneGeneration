# acceleration & misc setting
accurate_reproduce = False # 精确复现实验需开启
dtype = "fp32"  # fp16, bf16, fp32, 
grad_checkpoint = True
plugin = "zero2"
sp_size = 1
seed = 1234

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
batch_size = 1
num_workers = 1

# dataset settings
train_dataset = dict(
    type = 'DatasetForScanNetV2',
    data_root = '/mnt/data1/z00912387/scannetpp_pcd/train',
    data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train.txt',
    query_sample_rate = 0.1,
    max_query_number = 100000,
    Gridsample_mode = 'train',
    grid_size = 0.05,
)
val_dataset = dict(
    type = 'DatasetForScanNetV2',
    data_root = '/mnt/data1/z00912387/scannetpp_pcd/val',
    data_list_name = 'datacache/scannetpp_trainval/nvs_sem_val.txt',
    query_sample_rate = 0.1,
    max_query_number = 100000,
    Gridsample_mode = 'train',
    grid_size = 0.05,
)

# scheduler settings
epochs = 500
log_every = 1
ckpt_every = 500
eval_every = 500
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
loss_weight_kl = 0.05
# - eval metric
enable_occ_eval = False
eval_metric_list = ['precision','recall','f1','iou']
occ_threshold = 0.0

# Others
outputs = "./test/expr"



