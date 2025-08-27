import os
import warnings
import torch
from tqdm import tqdm
import numpy as np
import matplotlib
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import open3d as o3d

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRasterizer, 
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.preferred_linalg_library("magma")
warnings.filterwarnings("ignore")
matplotlib.use('Agg')  
DEBUG = False

def render_depth_infer(mesh, rasterizer_hr, batch_c2ws, hr_cam_params,scale, h_hr=1080, w_hr=1620):
    R_batch, T_batch = [], []              
    for idx, c2w in enumerate(batch_c2ws):
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        new_R = -R.T
        new_T = T.clone()
        new_T[:2] *= -1  # 翻转 x/y
        new_R[:, 2] *= -1  # 翻转 Z 轴

        R_batch.append(new_R)
        T_batch.append(new_T)
    R_batch = torch.stack(R_batch, dim=0)       # [B, 3, 3]
    T_batch = torch.stack(T_batch, dim=0)       # [B, 3]
    
    mesh_batch = mesh.extend(len(batch_c2ws))
    # 保存depth图
    # fx_hr, fy_hr, cx_hr, cy_hr = hr_cam_params               
    fx_hr = hr_cam_params[0,0]                                 
    fy_hr = hr_cam_params[1,1] 
    cx_hr = hr_cam_params[0,-1] 
    cy_hr = hr_cam_params[1,-1] 
    # print(f"h_hr: {h_hr}, w_hr: {w_hr}, fx_hr: {fx_hr}, fy_hr: {fy_hr}, cx_hr: {cx_hr}, cy_hr: {cy_hr}")                                                
    cameras_hr = PerspectiveCameras(
                                    focal_length=[(fx_hr, fy_hr)] * len(batch_c2ws),
                                    principal_point=[(cx_hr, cy_hr)] * len(batch_c2ws),                       
                                    image_size=[(h_hr, w_hr)] * len(batch_c2ws),
                                    R=R_batch,
                                    T=T_batch,
                                    in_ndc=False,
                                    device=device
                                )
    # print("cameras_hr world position:", cameras_hr.get_camera_center())                        
    fragments_hr = rasterizer_hr(mesh_batch, cameras=cameras_hr)
    depthmap_hr = fragments_hr.zbuf[..., 0]/scale
    depthmap_hr = depthmap_hr.detach().cpu()    
    return depthmap_hr  

def save_pack_as_pt(pack: dict, save_path: str):
    pack_tensor = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor):
            pack_tensor[k] = v
            print(f'{k}, {v.shape} \n')
        elif isinstance(v, np.ndarray):
            pack_tensor[k] = torch.from_numpy(v)
            print(f'{k}, {v.shape} \n')
        else:
            # 其他类型（字符串、数字、None等）直接保存
            pack_tensor[k] = v
            print(f'{k}, {v} \n')
    torch.save(pack_tensor, save_path)

def compute_absolute_error(depth_pred, depth_gt):
    return np.abs(depth_pred - depth_gt)

def compute_accuracy_within_threshold(abs_error, thresholds=[0.05, 0.1, 0.2]):
    return {t: np.mean(abs_error < t) for t in thresholds}

def compute_chamfer(pred_points, gt_points):
    """
    计算 Chamfer Distance
    Args:
        pred_points: numpy (N, 3)
        gt_points: numpy (M, 3)
    Returns:
        chamfer distance (float)
    """
    pred = torch.from_numpy(pred_points).float().unsqueeze(0).cuda()
    gt = torch.from_numpy(gt_points).float().unsqueeze(0).cuda()
    cd, _ = chamfer_distance(pred, gt)
    return cd.item()

if __name__ == '__main__':

    test_mode = 'udfgt' # ['hunyuan','udfgt','pred'] otherwise predict test

    ## input data
    gt_mesh_root = '/home/data/y00391175/project_xiaoxu/test_mesh_pipeline/test_mesh_gt/scannetpp_mesh_norm_-1_1_0.95/'
    gt_root = '/home/data/y00391175/project_xiaoxu/test_mesh_pipeline/test_mesh_gt/scannetpp_volume_norm_-1_1_0.95_render_depth/'

    if test_mode == 'udfgt':
        ## - test for udfgt
        test_out_root = '/home/data/y00391175/project_xiaoxu/test_mesh_pipeline/test_mesh_result/result_ours/'
        exp_name = 'exp0723_4en_8de_KL64_norm_sdf_baseline_1e-4_w10000_tsdf_vs256' # gt result folder
        prefix = 'eval_mode_tudf3_mcalpha1.5'    

    elif test_mode == 'hunyuan':
        ## - test for hunyuan
        test_out_root = '/home/data/y00391175/project_xiaoxu/test_mesh_pipeline/test_mesh_result/result_hunyuan3d-2/'
        exp_name = 'vs256' # gt result folder
        prefix = 'result_eval_scannet_align_ours'   
    else:
        ## -  test for ours
        test_out_root = '/home/data/y00391175/project_xiaoxu/test_mesh_pipeline/test_mesh_result/result_ours/'
        exp_name = 'exp0723_4en_8de_KL64_norm_sdf_baseline_1e-4_w10000_tsdf_vs256'
        # prefix = 'eval_mode_tudf3'
        prefix = 'eval_mode_tudf3_mcalpha1.5'

    pred_root = os.path.join(test_out_root,exp_name)

    ## output result
    if test_mode == 'udfgt':
        save_root = os.path.join(pred_root, f'{prefix}_render_depth_gt')
    else:     
        save_root = os.path.join(pred_root, f'{prefix}_render_depth')
    os.makedirs(save_root,exist_ok=True)

    ## get file list    
    with open("/home/data/y00391175/project_xiaoxu/scannetpp/splits/nvs_sem_val.txt") as f:
        sha256s = f.read().splitlines()

    ## depth render params
    rasterizer_hr = MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(
                image_size=(1080, 1620),
                blur_radius=0.0,
                faces_per_pixel=1,
                perspective_correct=True,  # 更快，无影响
            )
        )

    device = "cuda:3"
    vs_res = 256
    vs_size = 2.0/vs_res
    thresholds = [vs_size, 3.0*vs_size, 5.0*vs_size]
    USE_SAMPLE_POINTS = False
    N_POINTS = 50000
    SKIP_IF_DEPTH_EXISTS = True

    all_abs_errors_pred = []
    all_accs_pred = []
    all_cd_vals = []

    ##
    with torch.no_grad():
        for sha256 in tqdm(sha256s, desc="Rendering Depth"):
            save_dir = os.path.join(save_root, f'{sha256}')
            os.makedirs(save_dir,exist_ok=True)
            save_path = os.path.join(save_dir,f'{sha256}.pt')
            
            ## input_data
            input_data = torch.load(os.path.join(gt_root,f'{sha256}/{sha256}.pt')) # gt

            if test_mode == 'udfgt':
                input_mesh_file = os.path.join(pred_root,prefix,f'{sha256}/gt_mesh_{vs_res}.ply') # gt mesh
            elif test_mode == 'hunyuan':
                input_mesh_file = os.path.join(pred_root,prefix,f'{sha256}/{sha256}_{vs_res}.ply') # hunyuan mesh
            else:
                input_mesh_file = os.path.join(pred_root,prefix,f'{sha256}/pred_mesh_{vs_res}.ply') # pred mesh

            mesh_in =  o3d.io.read_triangle_mesh(input_mesh_file)
            input_mesh_file_gt = os.path.join(gt_mesh_root,f'{sha256}.ply')
            gt_mesh_in =  o3d.io.read_triangle_mesh(input_mesh_file_gt)
            depth_gt = input_data[f'depthmap_1080x1620_norm_128']

            if SKIP_IF_DEPTH_EXISTS and os.path.exists(save_path):
                print(f"[INFO] Skipping {sha256} (depth already rendered).")
                depthmap_hr = torch.load(save_path)[f'depthmap_1080x1620_norm_{vs_res}']
            else:
                ## input gt param
                batch_c2ws = input_data['extrinsics_Tnorm_norm_128'].to(device) # not related with vs_res
                hr_cam_params = input_data['resized_intrinsics_norm_128'].to(device)
                scale = torch.tensor(input_data['scale']).to(device)

                ## predict mesh
                vertices = torch.tensor(np.asarray(mesh_in.vertices), dtype=torch.float32).to(device)
                faces = torch.tensor(np.asarray(mesh_in.triangles), dtype=torch.int64).to(device)
                mesh = Meshes(verts=[vertices], faces=[faces])

                ## render
                depthmap_hr = render_depth_infer(mesh, rasterizer_hr, batch_c2ws, hr_cam_params, scale)
                print(f"[INFO] finished render depthmaps. Depthmap_hr: {depthmap_hr.shape, depthmap_hr.min(), depthmap_hr.max()}")
                
                ##
                pack = {}
                pack['scene_id'] = sha256  
                pack[f'depthmap_1080x1620_norm_{vs_res}'] = depthmap_hr
                save_pack_as_pt(pack, save_path)
                save_image(depthmap_hr.unsqueeze(1), os.path.join(save_dir, f'{sha256}_usedDepths.png'))

            ##
            ##- gt mesh & depth
            depth_pred = depthmap_hr.numpy()
            abs_err_pred = compute_absolute_error(depth_pred, depth_gt.numpy())
            acc_pred = compute_accuracy_within_threshold(abs_err_pred, thresholds=thresholds)

            ## cd metric
            if USE_SAMPLE_POINTS:
                gt_points = np.asarray(gt_mesh_in.sample_points_uniformly(number_of_points=N_POINTS).points)
                pred_points = np.asarray(mesh_in.sample_points_uniformly(number_of_points=N_POINTS).points)
            else:
                gt_points = np.asarray(gt_mesh_in.vertices)
                pred_points = np.asarray(mesh_in.vertices)

            if len(gt_points) == 0 or len(pred_points) == 0:
                cd_val = np.nan
                print(f"[WARN] Empty mesh in scene {sha256}, skipping Chamfer Distance.")
            else:
                cd_val = compute_chamfer(pred_points, gt_points)

            # result by scene
            scene_result_file = os.path.join(save_dir, f'results_{vs_res}.txt')
            with open(scene_result_file, "w") as f:
                f.write(f"Scene {sha256} results:\n")
                f.write(f"  [Pred]    MAE: {abs_err_pred.mean():.8f}")
                for t in thresholds:
                    f.write(f", Acc@{t:.2f}: {acc_pred[t]:.8f}")
                f.write(f"\n  Chamfer Distance: {cd_val:.8f}\n")

            print(f"\nScene {sha256} results:")
            print(f"  [Pred]    MAE: {abs_err_pred.mean():.8f}", end='')
            for t in thresholds:
                print(f", Acc@{t:.2f}: {acc_pred[t]:.8f}", end='')
            print(f"\n  Chamfer Distance: {cd_val:.8f}")

            all_abs_errors_pred.append(abs_err_pred.mean())
            all_accs_pred.append([acc_pred[t] for t in thresholds])
            all_cd_vals.append(cd_val)

    avg_mae = np.mean(all_abs_errors_pred)
    avg_accs = np.mean(all_accs_pred, axis=0)
    avg_cd = np.nanmean(all_cd_vals)
    

    ## write out the reuslt
    summary_file = os.path.join(save_root, f"all_scenes_mesh_eval_summary_{vs_res}.txt")
    with open(summary_file, "w") as f:
        f.write("=== All Scenes Average Evaluation ===\n")
        f.write(f"Average MAE: {avg_mae:.8f}\n")
        for i, t in enumerate(thresholds):
            f.write(f"Average Acc@{t:.2f}: {avg_accs[i]:.8f}\n")
        f.write(f"Average Chamfer Distance: {avg_cd:.8f}\n")

    print(f"Results saved to scene folders and summary saved to {summary_file}")
    print(f"\n=== All Scenes Average Evaluation ===")
    print(f"Average MAE: {avg_mae:.8f}")
    for i, t in enumerate(thresholds):
        print(f"Average Acc@{t:.2f}: {avg_accs[i]:8f}")
    print(f"Average Chamfer Distance: {avg_cd:.8f}")