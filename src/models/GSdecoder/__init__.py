from .decoder_RGBcnnCamDepthV0 import DecoderOnlyRGBCNNCamDepthInV0
from .decoder_RGBcnnCamDepthV1 import DecoderOnlyRGBCNNCamDepthInV1
from .decoder_RGBcnnCamDepthV2 import DecoderOnlyRGBCNNCamDepthInV2 # 修改分辨率为原图，load GPU权重
from .decoder_RGBcnnCamDepthV3 import DecoderOnlyRGBCNNCamDepthInV3 # 修改反投影
from .decoder_RGBcnnCamDepthV4 import DecoderOnlyRGBCNNCamDepthInV4 # 替换attention为serialized attn
from .decoder_RGBcnnCamDepthV5 import DecoderOnlyRGBCNNCamDepthInV5 # 基于V2，使用ori metric   
# from .decoder_RGBcnnCamDepthV2_saveMem import DecoderOnlyRGBCNNCamDepthInV2saveMem # 使用sparse Transformer
from .decoder_RGBcnnCamDepthV2_wmViT import DecoderOnlyInputViTV2 # 使用 worldmirror ViT 替换 CNN
from .decoder_RGBcnnCamDepthV4_unify import DecoderOnlyRGBCNNCamDepthInV4unify # 基于V2，使用ori metric   

