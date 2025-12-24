import os
import open3d as o3d
from tqdm import tqdm
import multiprocessing as mp

def sparse_voxelize(mesh, voxel_size=0.02):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    voxel_indices = [tuple(v.grid_index) for v in voxel_grid.get_voxels()]
    return set(voxel_indices)

def sparse_voxel_iou(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 1.0

def compute_sparse_iou(mesh1, mesh2, voxel_size=0.05):
    vox1 = sparse_voxelize(mesh1, voxel_size=voxel_size)
    vox2 = sparse_voxelize(mesh2, voxel_size=voxel_size)
    return sparse_voxel_iou(vox1, vox2)

def simplify_mesh(mesh, target_ratio=0.2):
    orig_tri_count = len(mesh.triangles)
    target_tri_count = int(orig_tri_count * target_ratio)
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tri_count)
    simplified.compute_vertex_normals()
    return simplified

def find_min_ratio_for_iou(mesh, iou_threshold=0.98, voxel_size=0.05):
    for ratio in [round(x, 2) for x in list(frange(0.1, 1.05, 0.05))]:
        simplified = simplify_mesh(mesh, target_ratio=ratio)
        iou = compute_sparse_iou(mesh, simplified, voxel_size=voxel_size)
        if iou >= iou_threshold:
            return ratio, simplified, iou
    return None, None, None

def frange(start, stop, step):
    while start <= stop:
        yield round(start, 5)
        start += step

# === 独立处理单个 scene 的函数 ===
def process_scene(args):
    scene_id, input_dir, output_dir, iou_threshold = args

    input_mesh_path = os.path.join(input_dir, scene_id, 'scans', 'mesh_aligned_0.05.ply')
    output_mesh_path = os.path.join(output_dir, scene_id, 'scans', 'mesh_aligned_0.05.ply')

    if not os.path.exists(input_mesh_path):
        return f"{scene_id}: MESH_NOT_FOUND"

    try:
        mesh = o3d.io.read_triangle_mesh(input_mesh_path)
        mesh.compute_vertex_normals()

        best_ratio, best_simplified, best_iou = find_min_ratio_for_iou(mesh, iou_threshold=iou_threshold)

        if best_simplified is not None:
            os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
            o3d.io.write_triangle_mesh(output_mesh_path, best_simplified)
            return f"{scene_id}: ratio={best_ratio:.2f}, IoU={best_iou:.4f}"
        else:
            return f"{scene_id}: FAILED_TO_MEET_IOU"
    except Exception as e:
        return f"{scene_id}: EXCEPTION - {str(e)}"

# === 主程序 ===
if __name__ == '__main__':
    iou_threshold = 0.98
    input_dir = '/data07/z00533370/dataset/scannetpp_mesh/data/'
    output_dir = '/data06/z00533370/dataset/scannetpp_mesh_simplify/data/'
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "simplify_log.txt")

    scene_ids = os.listdir(input_dir)
    args_list = [(scene_id, input_dir, output_dir, iou_threshold) for scene_id in scene_ids]

    print(f"[INFO] Launching parallel simplification on {len(scene_ids)} scenes...")

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_scene, args_list), total=len(scene_ids)))

    with open(log_file, 'w') as log:
        for line in results:
            log.write(line + '\n')
