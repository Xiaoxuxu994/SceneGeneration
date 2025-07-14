import torch
import torch_npu
import torch_fidelity

# metrics = torch_fidelity.calculate_metrics(
#     input1='./sample_outer',
#     input2='./vae_v1/scannetpp_kl-64_gen_outer',
#     fid=True,
#     kid=True,
#     cuda=False,
# )
# print(metrics)

metrics = torch_fidelity.calculate_metrics(
    input1='./mvr_path1',
    input2='./mvr_path2',
    fid=True,
    kid=True,
    cuda=False,
)
print(metrics)