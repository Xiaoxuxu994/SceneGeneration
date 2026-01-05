try:
    import torch_npu
    PLATFORM = "ASCEND"
    DEVICE_NAME = "npu"
except:
    PLATFORM = "NVIDIA"
    DEVICE_NAME = "cuda"

# PLATFORM = "NVIDIA"
# DEVICE_NAME = "cuda"