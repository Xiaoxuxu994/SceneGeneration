# from .refinerv1 import Refiner   # 单场景不owrk，舍弃
# from .refinerv2 import RefinerV2 # 最naive的版本，啥也么加
# from .refinerv3 import RefinerV3 # 增加了 delta_GS 的activation
# from .refinerv4 import RefinerV4 # 合入 feat voxel
# from .refinerv5 import RefinerV5 # 使用 PTV3 serialized Attention 替代 自己实现的knn attention
# # from .refinerv6 import RefinerV6 # refinerv5 基础上，改进knn attention block 的实现方式 为 PTV1
from .refinerv7 import RefinerV7 # refinerv5 基础上，减少显存
