# acceleration & misc setting
dtype = "fp32"  # fp16, bf16, fp32, 
grad_checkpoint = True
grad_checkpoint_step = 25
sp_size = 1
seed = 42

# model setting
model = dict(
    type="VecSetAEV4_SBMS",
    input_pcd_num = 50000,
    num_latent = 2048,
    latent_dim = 512,
    kl_latent_dim = 64,
    enable_kl = True,
    enable_ca_flash = False, # VAE cross attention
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
        type = "VecSetDecoder_SBMS", #['VecSetDecoder','VecSetKLDecoder']
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
)

# dataset (model.fps_method:0=V2, 1=V3, 2=V4)
train_dataset = dict(
    type="DatasetForScanNetV2",
    data_root='../../data/scannetpp_pcd/train',
    fps_root='../../data/scannetpp_pcd/train_fps_fix',
    data_list_name='datacache/scannetpp_trainval/nvs_sem_train_sub.txt',
    query_sample_rate= 0.1,
    max_query_number= 100000,
    Gridsample_mode = 'train',
    grid_size = 0.05,
)
val_dataset = dict(
    type="DatasetForScanNetV2",
    data_root='../../data/scannetpp_pcd/val',
    fps_root='../../data/scannetpp_pcd/val_fps_fix',
    data_list_name='datacache/scannetpp_trainval/nvs_sem_val.txt',
    query_sample_rate=0.1,
    max_query_number=100000,
    Gridsample_mode = 'train',
    grid_size = 0.05,
)   
# dataloader
batch_size = 6
num_workers = 1

# scheduler settings
epochs = 1
log_every = 1
eval_every = 2000
ckpt_every = 34750
resume = None
load = None

lr = 1e-4 
cos_total_iter = 5000 
cos_eta_min = 5e-7
enable_scaler = False
grad_clip = 1.0
weight_decay = 0


# loss & eval
# - occ geometry loss
enable_occ_loss = True
loss_weight_occ = 50
loss_weight_kl = 0.05
# - eval metric
enable_occ_eval = True
eval_metric_list = ['precision','recall','f1','iou']
occ_threshold = 0.0

# Others
outputs = "./output/expr-sbms"



