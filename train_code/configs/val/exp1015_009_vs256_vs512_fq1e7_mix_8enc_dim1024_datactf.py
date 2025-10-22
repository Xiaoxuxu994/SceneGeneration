# model setting
dim = 1024
model = dict(
    type="VecSetAEV4SINGLEKL",
    num_latent = 2048,
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
        dim_head = 64,
        enable_flash = True,
        attn_dtype = 'bf16',
        drop_path_rate = 0.3,
    ),
    decoder_dict4 = dict(
        type = "VecSetDecoderV1", 
        depth = 16,
        dim = dim,
        queries_dim = dim,
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
data_root = "/mnt/nvme1/share/datasets/scannetpp_norm_v3_02_vs16_256_oct512_3x/"
train_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train2.txt'
val_data_list_name = 'datacache/scannetpp_trainval/nvs_sem_val.txt'
vs_res = [256,512]
vs_res_gd = 256

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
    'ToTensor': {'enabled': True},
    'Collect':{
        'enabled':True,
        'keys':['coord','normal','scene_id',
        'pc_query_256','pc_query_grid_256','pc_query_sdf_256',
        'pc_query_512','pc_query_grid_512','pc_query_sdf_512']}
    }
)


#eval metric loss
truncatedist = [3 * 2.0 / 256, 3 * 2.0 / 256] #[3 * 2.0 / vs_res[-1]]
sdf_threshold = [2.0 / 256, 2.0 / 256] #[2.0 / vs_res[-1]]
mc_alpha = [1.0, 1.5]

# ckpt and output result
root = '/mnt/nvme3/z00533370/checkpoint/'
ckpt_name = 'exp1015/009_vs256_vs512_fq1e7_mix_8enc_dim1024/global_step112500/mp_rank_00_model_states.pt'
save_as_pt = False
outputs = "/mnt/nvme3/z00533370/test_result/"
prefix = f"datactf_{mc_alpha}_oct3x_debug"
