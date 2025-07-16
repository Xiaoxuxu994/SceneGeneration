#include <torch/extension.h>
#include <arm_neno.h>
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

void distance_squared_neon_fixed(
    const float* target,
    const float* points,  // 内存布局: [x0,y0,z0, x1,y1,z1,...]
    float* distances,
    size_t num_points)
{
    // 加载目标点坐标到向量寄存器
    float32x4_t target_x = vld1q_dup_f32(target);
    float32x4_t target_y = vld1q_dup_f32(target + 1);
    float32x4_t target_z = vld1q_dup_f32(target + 2);

    // 主循环：每次处理4个点
    for (size_t i = 0; i < num_points; i += 4) {
        // 加载4个点的坐标（需要手动收集非连续数据）
        float32x4_t x, y, z;
        for (int j = 0; j < 4 && (i + j) < num_points; ++j) {
            const float* p = points + (i + j) * 3;
            x[j] = p[0];
            y[j] = p[1];
            z[j] = p[2];
        }

        // 计算差值平方
        float32x4_t dx = vsubq_f32(x, target_x);
        float32x4_t dy = vsubq_f32(y, target_y);
        float32x4_t dz = vsubq_f32(z, target_z);
        
        float32x4_t dx2 = vmulq_f32(dx, dx);
        float32x4_t dy2 = vmulq_f32(dy, dy);
        float32x4_t dz2 = vmulq_f32(dz, dz);

        // 累加并存储结果
        float32x4_t sum = vaddq_f32(vaddq_f32(dx2, dy2), dz2);
        for (int j = 0; j < 4 && (i + j) < num_points; ++j) {
            distances[i + j] = sum[j];
        }
    }
}

void update_distances_neon(
    const float* last_point,      // [x0, y0, z0]
    const float* points,          // 连续存储的点云数据 [x1,y1,z1, x2,y2,z2,...]
    float* dists,
    const bool* selected_mask,    // 选择掩码数组
    int64_t batch_length)
{
    // 加载目标点到向量寄存器
    float32x4_t target_x = vld1q_dup_f32(last_point);
    float32x4_t target_y = vld1q_dup_f32(last_point + 1);
    float32x4_t target_z = vld1q_dup_f32(last_point + 2);
    
    // 每次处理4个点（NEON寄存器容量）
    int64_t p = 0;
    for (; p + 3 < batch_length; p += 4) {
        // 加载掩码（将bool转换为float32x4_t）
        uint32x4_t mask = vld1q_u32(reinterpret_cast<const uint32_t*>(&selected_mask[p]));
        uint32x4_t zero_mask = vceqq_u32(mask, vdupq_n_u32(0)); // 0xFFFFFFFF对应未选中的点
        
        // 仅处理未选中的点（selected_mask[p] == false）
        if (vgetq_lane_u32(zero_mask, 0) || vgetq_lane_u32(zero_mask, 1) || 
            vgetq_lane_u32(zero_mask, 2) || vgetq_lane_u32(zero_mask, 3)) {
            
            // 加载4个点的坐标（手动收集非连续数据）
            float32x4_t px, py, pz;
            px = vsetq_lane_f32(points[p*3 + 0], px, 0);
            py = vsetq_lane_f32(points[p*3 + 1], py, 0);
            pz = vsetq_lane_f32(points[p*3 + 2], pz, 0);
            
            px = vsetq_lane_f32(points[(p+1)*3 + 0], px, 1);
            py = vsetq_lane_f32(points[(p+1)*3 + 1], py, 1);
            pz = vsetq_lane_f32(points[(p+1)*3 + 2], pz, 1);
            
            px = vsetq_lane_f32(points[(p+2)*3 + 0], px, 2);
            py = vsetq_lane_f32(points[(p+2)*3 + 1], py, 2);
            pz = vsetq_lane_f32(points[(p+2)*3 + 2], pz, 2);
            
            px = vsetq_lane_f32(points[(p+3)*3 + 0], px, 3);
            py = vsetq_lane_f32(points[(p+3)*3 + 1], py, 3);
            pz = vsetq_lane_f32(points[(p+3)*3 + 2], pz, 3);
            
            // 计算差值平方
            float32x4_t dx = vsubq_f32(px, target_x);
            float32x4_t dy = vsubq_f32(py, target_y);
            float32x4_t dz = vsubq_f32(pz, target_z);
            
            float32x4_t dx2 = vmulq_f32(dx, dx);
            float32x4_t dy2 = vmulq_f32(dy, dy);
            float32x4_t dz2 = vmulq_f32(dz, dz);
            
            // 计算距离平方
            float32x4_t new_dist = vaddq_f32(vaddq_f32(dx2, dy2), dz2);
            
            // 加载当前距离值
            float32x4_t current_dist = vld1q_f32(&dists[p]);
            
            // 条件更新：仅当(new_dist < current_dist)且(!selected_mask)时更新
            uint32x4_t update_mask = vandq_u32(
                vcltq_f32(new_dist, current_dist),
                zero_mask
            );
            
            // 混合新旧值
            float32x4_t updated_dist = vbslq_f32(update_mask, new_dist, current_dist);
            vst1q_f32(&dists[p], updated_dist);
        }
    }
    
    // 处理剩余不足4个的点（标量处理）
    for (; p < batch_length; ++p) {
        if (!selected_mask[p]) {
            float dx = points[p*3] - last_point[0];
            float dy = points[p*3 + 1] - last_point[1];
            float dz = points[p*3 + 2] - last_point[2];
            float new_dist = dx*dx + dy*dy + dz*dz;
            
            if (new_dist < dists[p]) {
                dists[p] = new_dist;
            }
        } else {
            dists[p] = 0.0f;
        }
    }
}


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
        // #pragma omp simd
        // for (int64_t p = 0; p < batch_length; ++p) {
        //     dists[p] = distance_squared_3d(last_point, &points_a[n][p][0]);
        // }
        distance_squared_neon_fixed(last_point, &points_a[n][0][0], dists.data(), batch_length);

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
            update_distances_neon(
                last_point,                // 当前采样点坐标
                &points_a[n][0][0],        // 点云数据起始指针
                dists.data(),              // 距离数组
                selected_points_mask.data(), // 选择掩码
                batch_length               // 点数量
            );
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
        distance_squared_neon_fixed(last_point, &points_a[n][0][0], dists.data(), batch_length);

        // 迭代采样剩余点
        for (int64_t k = 1; k < batch_k; ++k) {
            // 使用KD-Tree进行最近邻搜索优化
            std::vector<nanoflann::ResultItem<size_t, float>> ret_matches;  // 使用nanoflann的正确返回类型

            // 查找距离当前点最远的未选择点
            size_t farthest_idx = 0;
            float max_dist = 0.0f;
            
            #pragma omp simd reduction(max:max_dist)
            for (int64_t p = 0; p < batch_length; ++p) {
                if (!selected_points_mask[p] &&
                     dists[p] > max_dist) {
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
            kdtree.radiusSearch(
                new_point, 
                radius, 
                ret_matches, 
                nanoflann::SearchParameters(0.05, false)
            );

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