from .model_v2 import PointTransformerV3_2
import torch 

data_path = '/home/y00391175/project_xiaoxu/dataset/3DGS/scannetpp_sample/data/ef25276c25_process/train/ef25276c25_transform_feat_coord_sample50000.pth'
input_dict = torch.load(data_path)

model_ptv3 = PointTransformerV3_2(
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2,),
        enc_depths=(1, 1,),
        enc_channels=(32, 64,),
        enc_num_head=(2, 4,),
        enc_patch_size=(1024, 1024,),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        attn_dtype='fp16',
        enable_ca_flash=False,
        cls_mode=True,
    ).cuda()

print(' ============== model_ptv3 =====================')
print(model_ptv3)


for key in input_dict.keys():
    if isinstance(input_dict[key], torch.Tensor):
        input_dict[key] = input_dict[key].cuda(non_blocking=True)

fea_enc = model_ptv3(input_dict)
print(fea_enc)