import torch
import torch_npu

import numpy as np
import open3d as o3d

import imageio
from PIL import Image
from pathlib import Path

CKPT = "../../checkpoints/VGGT-1B/"
NUM_FRAMES = 10
INTERVAL = 40
ITERATOR = range(0, INTERVAL * NUM_FRAMES, INTERVAL)
IMAGES = [
    "../../samples/7scenes/data/chess/seq-01/frame-{}.color.png".format(str(s).zfill(6)) for s in ITERATOR
]
POSES = [
    "../../samples/7scenes/data/chess/seq-01/frame-{}.pose.txt".format(str(s).zfill(6)) for s in ITERATOR
] # camera to world 4x4 homogeneous matrices
DEPTHS = [
    "../../samples/7scenes/data/chess/seq-01/frame-{}.depth.proj.png".format(str(s).zfill(6)) for s in ITERATOR
    # "../../samples/7scenes/data/chess/seq-01/frame-{}.depth.png".format(str(s).zfill(6)) for s in ITERATOR
]

OUTPUT_ROOT = Path("./outputs/7scenes/sample_seq/")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("npu:0")
DTYPE = torch.float

# NOTE this is hard-coded and kept identical to pi3
# please check https://github.com/yyfz/Pi3/blob/evaluation/datasets/sevenscenes.py#L130
FX, FY, CX, CY = [525, 525, 320, 240] 

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from vggt.utils.metric import se3_to_relative_pose_error, calculate_auc, umeyama, accuracy, completion
from vggt.utils.camera import export_cam_as_ply
from training.data.dataset_util import depth_to_world_coords_points
from training.data.cropping import resize_image_depth_and_intrinsic
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch

def load_model(ckpt=CKPT):
    model = VGGT.from_pretrained(ckpt)
    model.eval()
    
    return model

def load_rgb(image_path_list=IMAGES):
    image_pils = []
    for image_path in image_path_list:
        image = Image.open(image_path)
        image_pils.append(image)
        
    return image_pils, load_and_preprocess_images(image_path_list) 

def filter_depth(depthmaps, far=10, near=1e-3):
    depthmaps[depthmaps > far] = 0
    depthmaps[depthmaps < near] = 0
    
    depth_masks = depthmaps != 0
    
    return depthmaps, depth_masks

def load_depth(depth_path_list=DEPTHS, invalid_depth=60000):
    depths = []
    for depth_path in depth_path_list:
        depth = imageio.v2.imread(depth_path).astype(np.float32)
        depths.append(depth)
        
    depths = np.stack(depths)
    depths = torch.from_numpy(depths)
    depths[depths >= invalid_depth] = 0
    depths /= 1000.0
    
    return depths

def load_extrinsic(pose_path_list=POSES): # load the world-to-camera matrices
    
    def parse_pose(pose_path): # get the camera-to-world matrices
        with open(pose_path) as f:
            lines = []
            for line in f.readlines():
                line = [float(l.lstrip().rstrip()) for l in line.lstrip().rstrip().split("\t")]
                lines.append(line)
        
        c2w = np.array(lines)
        
        return c2w
    
    c2w_arr = []
    for pose_path in pose_path_list:
        c2w = parse_pose(pose_path)
        c2w_arr.append(c2w)
        
    c2w_arr = torch.from_numpy(np.stack(c2w_arr)) # S, 4, 4
    w2c_arr = closed_form_inverse_se3(c2w_arr) # NOTE this is NOT lossless inversion
    
    return w2c_arr

def load_intrinsic(fx=FX, fy=FY, cx=CX, cy=CY):
    mat = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    mat = torch.from_numpy(mat).reshape(1, 3, 3).repeat(NUM_FRAMES, 1, 1)
    
    return mat

image_pils, images = load_rgb()
depths = load_depth()
gt_intrinsics = load_intrinsic()
gt_w2c = load_extrinsic()

def build_gt_geometry_data(image_pils, depths, raw_extrinsics, raw_intrinsics):
    S = depths.shape[0]
    
    new_depthmaps, new_intrinsics, extrinsics_3x4, cam_points, world_points = [], [], [], [], []
    for i in range(S):
        image_pil = image_pils[i]
        depth_map = depths[i].numpy()
        extrinsic = raw_extrinsics[i, :3].numpy() # 3, 4
        intrinsic = raw_intrinsics[i].numpy() # 3, 3
        
        _, depth_map, intrinsic = resize_image_depth_and_intrinsic(
            image=image_pil,
            depth_map=depth_map,
            intrinsic=intrinsic,
            output_width=518
        )
        
        (
            world_coords_points, 
            cam_coords_points, 
            _
        ) = depth_to_world_coords_points(depth_map, extrinsic, intrinsic)
        
        new_depthmaps.append(depth_map)
        new_intrinsics.append(intrinsic)
        extrinsics_3x4.append(extrinsic)
        cam_points.append(cam_coords_points)
        world_points.append(world_coords_points)
        
    new_depthmaps = np.stack(new_depthmaps)
    new_intrinsics = np.stack(new_intrinsics)
    extrinsics_3x4 = np.stack(extrinsics_3x4)
    cam_points = np.stack(cam_points)
    world_points = np.stack(world_points)
    
    new_depthmaps, depth_masks = filter_depth(new_depthmaps)
        
    (
        new_extrinsics, 
        new_cam_points, 
        new_world_points, 
        new_depthmaps
    ) = normalize_camera_extrinsics_and_points_batch(
        extrinsics=torch.from_numpy(extrinsics_3x4).unsqueeze(0),
        cam_points=torch.from_numpy(cam_points).unsqueeze(0),
        world_points=torch.from_numpy(world_points).unsqueeze(0),
        depths=torch.from_numpy(new_depthmaps).unsqueeze(0),
        point_masks=torch.from_numpy(depth_masks).unsqueeze(0),
    )
    
    return new_intrinsics, new_extrinsics, new_cam_points, new_world_points, new_depthmaps, depth_masks

(
    gt_intrinsics, 
    gt_extrinsics, 
    gt_cam_points, 
    gt_point_maps, 
    gt_depth_maps,
    gt_depth_masks
) = build_gt_geometry_data(image_pils, depths, gt_w2c, gt_intrinsics)


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(gt_point_map.reshape(-1, 3))
# o3d.io.write_point_cloud("vggt_pointmap_gt.ply", pcd)
# export_cam_as_ply("vggt_pose_gt.ply", closed_form_inverse_se3(gt_extrinsics.squeeze(0)).unsqueeze(0))
        
# exit()

model = load_model().to(DEVICE)

with torch.no_grad():
    with torch.autocast(device_type=images.device.type, dtype=DTYPE):
        images = images[None].to(DEVICE)
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        (
            pred_extrinsics, # B, S, 4, 4 - SE(3)
            pred_intrinsics  # B, S, 3, 3
        ) = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:], is_se3=True)
    
        # Predict Depth Maps
        pred_depth_maps, pred_depth_confs = model.depth_head(aggregated_tokens_list, images, ps_idx)
    
        # Predict Point Maps
        pred_point_maps, pred_point_confs = model.point_head(aggregated_tokens_list, images, ps_idx)
    
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        pred_point_maps_by_unprojection = unproject_depth_map_to_point_map(pred_depth_maps.squeeze(0), 
                                                                    pred_extrinsics.squeeze(0), 
                                                                    pred_intrinsics.squeeze(0))
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pred_point_map_by_unprojection.reshape(-1, 3))
        # o3d.io.write_point_cloud("vggt_pointmap_pred.ply", pcd)
        # export_cam_as_ply("vggt_pose_pred.ply", closed_form_inverse_se3(pred_extrinsics.squeeze(0)).unsqueeze(0))
        
        # exit()
        
        
    # pose evaluation
    # NOTE to-be-evaluated poses must be world-from-camera matrices
    pred_extrinsics = closed_form_inverse_se3(pred_extrinsics.squeeze(0))
    pred_extrinsics = pred_extrinsics.to("cpu") # B = 1
    gt_extrinsics = closed_form_inverse_se3(gt_extrinsics.squeeze(0))
    gt_extrinsics = gt_extrinsics.to("cpu") # B = 1
    
    # save predictions
    pred_extrinsics = pred_extrinsics.numpy()
    pred_depth_maps = pred_depth_maps.squeeze(0).to("cpu").numpy()
    pred_point_maps = pred_point_maps.squeeze(0).to("cpu").numpy()

    print(pred_extrinsics.shape)
    print(pred_depth_maps.shape)
    print(pred_point_maps.shape)
    print(pred_point_maps_by_unprojection.shape)
    np.save(OUTPUT_ROOT / "pred_extrinsics.npy", pred_extrinsics)
    np.save(OUTPUT_ROOT / "pred_depth_maps.npy", pred_depth_maps)
    np.save(OUTPUT_ROOT / "pred_point_maps.npy", pred_point_maps)
    np.save(OUTPUT_ROOT / "pred_point_maps_by_unprojection.npy", pred_point_maps_by_unprojection)
    
    # save GTs
    gt_extrinsics = gt_extrinsics.numpy()
    gt_depth_maps = gt_depth_maps.squeeze(0).to("cpu").numpy()
    gt_point_maps = gt_point_maps.squeeze(0).to("cpu").numpy()
    
    print(gt_extrinsics.shape)
    print(gt_depth_maps.shape)
    print(gt_point_maps.shape)
    print(gt_depth_masks.shape)
    np.save(OUTPUT_ROOT / "gt_extrinsics.npy", gt_extrinsics)
    np.save(OUTPUT_ROOT / "gt_depth_maps.npy", gt_depth_maps)
    np.save(OUTPUT_ROOT / "gt_point_maps.npy", gt_point_maps)
    np.save(OUTPUT_ROOT / "gt_depth_masks.npy", gt_depth_masks)
    
    # THRESHOLD = 30
    # rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_extrinsics, gt_extrinsics, NUM_FRAMES)
    # rra_acc = (rel_rangle_deg <= THRESHOLD).sum() / rel_rangle_deg.shape[0]
    # rta_acc = (rel_tangle_deg <= THRESHOLD).sum() / rel_tangle_deg.shape[0]
    # auc_at_threshold = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=THRESHOLD).item()
    # print(f"Threshold: {THRESHOLD}, RRA: {rra_acc}, RTA: {rta_acc}, AUC: {auc_at_threshold}")
    
    # # point map evaluation
    # # coarse align
    # pred_pts = pred_point_map_by_unprojection
    # gt_pts = gt_point_map.squeeze(0).to("cpu").numpy()
    # valid_mask = gt_depth_masks

    # c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
    # # pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T
    
    # S, H, W, _ = pred_pts.shape
    # pred_pts = pred_pts.reshape((S, -1, 3))
    # R = np.repeat(R[None, :], S, axis=0)
    # t = np.repeat(t[None, :], S, axis=0)

    # pred_pts = pred_pts @ R
    # pred_pts = pred_pts + np.transpose(t, (0, 2, 1))
    # pred_pts = pred_pts.reshape((S, H, W, 3))
    
    # # 6. filter invalid points
    # pred_pts = pred_pts[valid_mask].reshape(-1, 3)
    # gt_pts = gt_pts[valid_mask].reshape(-1, 3)

    # # 7. save predicted & ground truth point clouds
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pred_pts)

    # pcd_gt = o3d.geometry.PointCloud()
    # pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)

    # # 8. ICP align refinement
    # threshold = 0.1

    # trans_init = np.eye(4)
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pcd,
    #     pcd_gt,
    #     threshold,
    #     trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # )

    # transformation = reg_p2p.transformation
    # pcd = pcd.transform(transformation)
    
    # # 9. estimate normals
    # pcd.estimate_normals()
    # pcd_gt.estimate_normals()
    # pred_normal = np.asarray(pcd.normals)
    # gt_normal = np.asarray(pcd_gt.normals)

    # # o3d.io.write_point_cloud(
    # #     os.path.join(
    # #         save_path, f"{seq.replace('/', '_')}-mask-icp.ply"
    # #     ),
    # #     pcd,
    # # )

    # # 10. compute metrics
    # acc, acc_med, nc1, nc1_med = accuracy(
    #     pcd_gt.points, pcd.points, gt_normal, pred_normal
    # )
    # comp, comp_med, nc2, nc2_med = completion(
    #     pcd_gt.points, pcd.points, gt_normal, pred_normal
    # )
    # print(
    #     f"Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
    # )