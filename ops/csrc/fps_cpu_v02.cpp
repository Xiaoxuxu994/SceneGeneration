#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <nanoflann.hpp>

// 3D点云适配器
struct PointCloudAdapter {
    const float* data;
    size_t num_points;

    inline size_t kdtree_get_point_count() const { return num_points; }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return data[idx * 3 + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// 针对3D点云优化的距离平方计算
inline float distance_squared_3d(const float* a, const float* b) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return dx*dx + dy*dy + dz*dz;
}

// 使用AVX优化的3D距离计算（虽然3D不能完全利用AVX，但比标量计算快）
#ifdef __AVX2__
inline float avx_distance_squared_3d(const float* a, const float* b) {
    // 加载3个元素，第4个元素设为0
    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 diff = _mm_sub_ps(va, vb);
    __m128 sq = _mm_mul_ps(diff, diff);
    
    // 水平相加：x+y+z+0
    __m128 shuf = _mm_movehdup_ps(sq);        // y, y, w, w
    __m128 sums = _mm_add_ps(sq, shuf);      // x+y, y+y, z+w, w+w
    shuf = _mm_movehl_ps(shuf, sums);        // z+w, w+w, x+y, y+y
    sums = _mm_add_ss(sums, shuf);           // x+y+z+w, ...
    
    return _mm_cvtss_f32(sums);
}
#else
inline float avx_distance_squared_3d(const float* a, const float* b) {
    return distance_squared_3d(a, b); // 回退到普通实现
}
#endif


torch::Tensor farthest_point_sampling_cpu_optimized(
    const torch::Tensor& points,
    const torch::Tensor& lengths,
    const torch::Tensor& K,
    const torch::Tensor& start_idxs) {
  
    // 输入验证
    TORCH_CHECK(points.dim() == 3, "points must be 3D tensor");
    TORCH_CHECK(points.size(2) == 3, "points must have 3 dimensions (x,y,z)");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D tensor");
    TORCH_CHECK(K.dim() == 1, "K must be 1D tensor");
    TORCH_CHECK(start_idxs.dim() == 1, "start_idxs must be 1D tensor");
    TORCH_CHECK(points.size(0) == lengths.size(0), "batch size mismatch");
    
    const int64_t N = points.size(0);
    const int64_t P = points.size(1);
    const int64_t max_K = torch::max(K).item<int64_t>();

    // 输出张量 (N, max_K)
    auto long_opts = lengths.options();
    torch::Tensor sampled_indices = torch::full({N, max_K}, -1, long_opts);

    // 访问器
    auto points_a = points.accessor<float, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto k_a = K.accessor<int64_t, 1>();
    auto sampled_indices_a = sampled_indices.accessor<int64_t, 2>();
    auto start_idxs_a = start_idxs.accessor<int64_t, 1>();

    // 为每个batch并行处理
    
    for (int64_t n = 0; n < N; ++n) {
        const int64_t batch_length = lengths_a[n];
        const int64_t batch_k = std::min(batch_length, k_a[n]);
        
        // 使用vector<bool>更节省空间
        std::vector<bool> selected_points_mask(batch_length, false);
        std::vector<float> dists(batch_length, std::numeric_limits<float>::max());

        // 初始点
        int64_t last_idx = start_idxs_a[n];
        sampled_indices_a[n][0] = last_idx;
        selected_points_mask[last_idx] = true;

        // 预加载上一个点的坐标
        const float* last_point = &points_a[n][last_idx][0];

        // 计算初始距离
        #pragma omp simd
        for (int64_t p = 0; p < batch_length; ++p) {
            dists[p] = distance_squared_3d(last_point, &points_a[n][p][0]);
        }

        // 迭代采样剩余点
        for (int64_t k = 1; k < batch_k; ++k) {
            // 找到最远点
            int64_t farthest_idx = std::distance(
                dists.begin(), 
                std::max_element(dists.begin(), dists.end()));
            
            sampled_indices_a[n][k] = farthest_idx;
            selected_points_mask[farthest_idx] = true;
            
            // 更新last_point
            last_point = &points_a[n][farthest_idx][0];
            
            // 更新距离
            #pragma omp simd
            for (int64_t p = 0; p < batch_length; ++p) {
                if (!selected_points_mask[p]) {
                    float new_dist = distance_squared_3d(last_point, &points_a[n][p][0]);
                    if (new_dist < dists[p]) {
                        dists[p] = new_dist;
                    }
                } else {
                    dists[p] = 0.0f;
                }
            }
        }
    }
    
    return sampled_indices;
}

torch::Tensor farthest_point_sampling_cpu_kdtree(
    const torch::Tensor& points,
    const torch::Tensor& lengths,
    const torch::Tensor& K,
    const torch::Tensor& start_idxs) {
  
    // 输入验证
    TORCH_CHECK(points.dim() == 3, "points must be 3D tensor");
    TORCH_CHECK(points.size(2) == 3, "points must have 3 dimensions (x,y,z)");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D tensor");
    TORCH_CHECK(K.dim() == 1, "K must be 1D tensor");
    TORCH_CHECK(start_idxs.dim() == 1, "start_idxs must be 1D tensor");
    TORCH_CHECK(points.size(0) == lengths.size(0), "batch size mismatch");
    
    const int64_t N = points.size(0);
    const int64_t P = points.size(1);
    const int64_t max_K = torch::max(K).item<int64_t>();

    // 输出张量 (N, max_K)
    auto long_opts = lengths.options();
    torch::Tensor sampled_indices = torch::full({N, max_K}, -1, long_opts);

    // 访问器
    auto points_a = points.accessor<float, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto k_a = K.accessor<int64_t, 1>();
    auto sampled_indices_a = sampled_indices.accessor<int64_t, 2>();
    auto start_idxs_a = start_idxs.accessor<int64_t, 1>();

    // 为每个batch并行处理
    
    for (int64_t n = 0; n < N; ++n) {
        const int64_t batch_length = lengths_a[n];
        const int64_t batch_k = std::min(batch_length, k_a[n]);
        
        // 初始化KD-Tree
        PointCloudAdapter pc;
        pc.data = &points_a[n][0][0];
        pc.num_points = batch_length;

        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, PointCloudAdapter>,
            PointCloudAdapter,
            3,
            size_t>;

        KDTree kdtree(3, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kdtree.buildIndex();
        
        // 使用vector<bool>更节省空间
        std::vector<bool> selected_points_mask(batch_length, false);
        std::vector<float> dists(batch_length, std::numeric_limits<float>::max());

        // 初始点
        int64_t last_idx = start_idxs_a[n];
        sampled_indices_a[n][0] = last_idx;
        selected_points_mask[last_idx] = true;

        // 预加载上一个点的坐标
        const float* last_point = &points_a[n][last_idx][0];

        // 计算初始距离
        #pragma omp simd
        for (int64_t p = 0; p < batch_length; ++p) {
            dists[p] = distance_squared_3d(last_point, &points_a[n][p][0]);
        }

        // 迭代采样剩余点
        for (int64_t k = 1; k < batch_k; ++k) {
            // 使用KD-Tree进行最近邻搜索优化
            std::vector<nanoflann::ResultItem<size_t, float>> ret_matches;  // 使用nanoflann的正确返回类型
            nanoflann::SearchParameters params;
            params.sorted = false;

            // 查找距离当前点最远的未选择点
            size_t farthest_idx = 0;
            float max_dist = 0.0f;
            
            #pragma omp simd reduction(max:max_dist)
            for (int64_t p = 0; p < batch_length; ++p) {
                if (!selected_points_mask[p] && dists[p] > max_dist) {
                    max_dist = dists[p];
                    farthest_idx = p;
                }
            }

            // 更新选择点和距离
            sampled_indices_a[n][k] = farthest_idx;
            selected_points_mask[farthest_idx] = true;
            const float* new_point = &points_a[n][farthest_idx][0];

            // 使用KD-Tree进行半径搜索优化距离更新
            const float radius = std::sqrt(max_dist);
            kdtree.radiusSearch(new_point, radius, ret_matches, params);

            #pragma omp simd
            for (size_t i = 0; i < ret_matches.size(); ++i) {
                const size_t p = ret_matches[i].first;
                if (!selected_points_mask[p]) {
                    const float new_dist = ret_matches[i].second;
                    if (new_dist < dists[p]) {
                        dists[p] = new_dist;
                    }
                }
            }
            // 处理未被半径搜索覆盖的点
            #pragma omp simd
            for (int64_t p = 0; p < batch_length; ++p) {
                if (selected_points_mask[p]) {
                    dists[p] = 0.0f;
                }
            }
        }
    }
    
    return sampled_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling_cpu_small", &farthest_point_sampling_cpu_optimized,
          "Optimized FPS for 3D point clouds");
    m.def("farthest_point_sampling_cpu_large", &farthest_point_sampling_cpu_kdtree,
          "Optimized FPS for 3D point clouds");
}