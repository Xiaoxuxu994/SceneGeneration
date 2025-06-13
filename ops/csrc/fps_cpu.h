#include <torch/extension.h>

torch::Tensor farthest_point_sampling_cpu(
    const torch::Tensor& points,
    const torch::Tensor& lengths,
    const torch::Tensor& K,
    const torch::Tensor& start_idxs);