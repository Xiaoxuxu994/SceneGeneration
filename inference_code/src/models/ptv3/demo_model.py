from src.models.ptv3.model_v2 import PointTransformerV3_2
import torch 
import torch_npu
from src.dataset.transform import *

data_path = '/mnt/nvme3/z00533370/dataset/scannetpp_trainval_v6/val/09c1414f1b.pth'
input_dict = torch.load(data_path)
gridsample = GridSample(
    grid_size = 0.05,
    hash_type = 'fnv',
    mode = 'train',
    keys = ("coord",),
    return_grid_coord = True,)
DecodeQuerySample = DecodeQuerySample(
                sample_rate = 0.1,
                max_number = 100000 ,
                mode = 'train',
                prefix = '0.05', )
toTensor = ToTensor()
collect = Collect(
    keys = ("coord", "grid_coord", "pc_query", "pc_query_label", "scene_id"),
    feat_keys = ("coord",)
    )
input_dict = gridsample(input_dict)
input_dict = DecodeQuerySample(input_dict)
input_dict = toTensor(input_dict)
input_dict = collect(input_dict)
device = torch.device("npu:0")

model_ptv3 = PointTransformerV3_2(
        in_channels=3,
        stride=(),
        enc_depths=(1,),
        enc_channels=(512, ),
        enc_num_head=(32, ),
        enc_patch_size=(1024, ),
        enable_cpe_block=False,
        enable_flash=True,
        attn_dtype='bf16',
        enable_ca_flash=False,
        drop_path=0.3,
        shuffle_orders=True,
        cls_mode=True, #True for enc only; False for enc-dec backbone
    ).to(device)

print(' ============== model_ptv3 =====================')
print(model_ptv3)


for key in input_dict.keys():
    if isinstance(input_dict[key], torch.Tensor):
        input_dict[key] = input_dict[key].to(device)

fea_enc = model_ptv3(input_dict)
print(fea_enc.feat)