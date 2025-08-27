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
## eval data
data_root = '/mnt/nvme1/share/datasets/scannetpp_mesh_norm_-1_1_0.95/'
val_data_list_name = 'datacache/scannetpp_trainval/test.txt'
vs_res = [256]
truncatedist = 3 * 2.0 / vs_res[-1]
test_data_args = dict(
    vs_res = vs_res,
    transform = {
    'GridSample': {
        'enabled': True,
        'grid_size': 2.0 / vs_res[-1], # encoder voxelsize
        'hash_type': 'fnv',
        'mode': 'train', # train for both train and test
        'keys': ('coord',),
        'return_grid_coord': False
    },
    'ToTensor': {'enabled': True},
    # 'Collect':{
    #     'enabled':True,
    #     'keys':['coord','pc_query_256','pc_query_sdf_256','scene_id']}
    }
)

#eval metric loss
sdf_threshold = 2.0 / vs_res[-1]
mc_alpha = 1.0

# ckpt and output result
root = '/mnt/nvme3/z00533370/checkpoint/'
ckpt_name = 'exp0723_4en_8de_KL64_norm_sdf_baseline_1e-4_w10000_tsdf_vs256/epoch495-global_step103000/model_ckpt.pt'
save_as_pt = True