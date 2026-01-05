import os
from argparse import ArgumentParser
from tqdm import tqdm
import json
import dotted_dict
import numpy as np

from utils.config_loader import LoadConfig

PROCESS_TOTAL = 0
PROCESS_IDX = 7

def is_frame_complete(scene_folder: str, frame_meta: dict):
    """Check the completeness of acquired data for a specific frame.

    Args:
        scene_folder (str): Absolute path to the scene folder.
        frame_meta (dict): Meta data of the frame to be checked.
    
    Returns:
        bool: Whether or not the frame is valid.
    """
    
    image_path = os.path.join(scene_folder, frame_meta["image"])
    moge2_mask_path = os.path.join(scene_folder, frame_meta["moge2_mask"])
    mvsa_depth_path = os.path.join(scene_folder, frame_meta["mvsanywhere_depth"])
    mvsa_depth_conf_path = os.path.join(scene_folder, frame_meta["mvsanywhere_depth_confidence"])
    
    if (not os.path.exists(image_path) or 
        not os.path.exists(moge2_mask_path) or 
        not os.path.exists(mvsa_depth_path) or 
        not os.path.exists(mvsa_depth_conf_path)
        ):
        return False
    return True


def is_scene_complete(scene_folder: str, scene_meta: dict):
    """Check the complete ness of acquired data for a specific scene.

    Args:
        scene_folder (str): Absolute path to the scene folder.
        scene_meta (dict): Meta data of the frame to be check.

    Returns:
        bool: Whether or not the scene is valid.
    """
    frame_metas = scene_meta["frames"]
    for frame_idx, frame_meta in enumerate(frame_metas):
        if frame_meta["colmap_im_id"] != frame_idx + 1:
            return False            
        cam_pose = np.array(frame_meta["transform_matrix"])
        if np.max(np.abs(cam_pose[:3, 3])) > 100:
            return False
    
    image_folder = os.path.join(scene_folder, "images")
    moge_mask_folder = os.path.join(scene_folder, "moge/v0/mask/moge2")
    mvsa_depth_folder = os.path.join(scene_folder, "mvsanywhere/v0/depth")
    mvsa_depth_conf_folder = os.path.join(scene_folder, "mvsanywhere/v0/depth_confidence")
    
    image_files = os.listdir(image_folder)
    moge_mask_files = os.listdir(moge_mask_folder)
    mvsa_depth_files = os.listdir(mvsa_depth_folder)
    mvsa_depth_conf_files = os.listdir(mvsa_depth_conf_folder)
    
    return (len(frame_metas) == len(image_files) and
            len(image_files) == len(moge_mask_files) and 
            len(image_files) == len(mvsa_depth_files) and 
            len(image_files) == len(mvsa_depth_conf_files))


def main(cfg: dotted_dict):
    print("########################### Start checking DL3DV_WAI format completeness ###########################")
        
    all_scene_names = sorted(os.listdir(cfg.datasets.dl3dv_wai.data_root))
    print("[ INFO ] {} folders found in root directory.".format(len(all_scene_names)))
    if PROCESS_TOTAL > 0:
        all_scene_names = all_scene_names[PROCESS_IDX::PROCESS_TOTAL]
        print("[ INFO ] Using multi-process for preprocess, current process {}/{}, {} scenes distributed.".format(
            PROCESS_IDX, PROCESS_TOTAL, len(all_scene_names)))
    
    invalid_scenes = []
    valid_scenes = []
    for scene_name in tqdm(all_scene_names, desc="Checking data completeness of DL3DV_WAI dataset"):
        abs_scene_path = os.path.join(cfg.datasets.dl3dv_wai.data_root, scene_name)
        try:
            with open(os.path.join(abs_scene_path, "scene_meta.json")) as fp:
                scene_meta = json.load(fp)
        except:
            invalid_scenes.append(scene_name)
            print("[ WARNING ] Invalid scene found ({}) for meta data not exist, invalid count: {}".format(
                scene_name, len(invalid_scenes)))
            continue
        if not is_scene_complete(abs_scene_path, scene_meta):
            invalid_scenes.append(scene_name)
            print("[ WARNING ] Invalid scene found ({}) for scene content not complete, invalid count: {}".format(
                scene_name, len(invalid_scenes)))
        else:
            valid_scenes.append(scene_name)
    
    print("[ INFO ] Data validation complete, {}/{} scenes found invalid.".format(len(invalid_scenes), len(all_scene_names)))
    output_file_name = "./dl3dv_wai_train_list.json" if PROCESS_TOTAL <= 0 else "./dl3dv_wai_train_list{}.json".format(PROCESS_IDX)
    with open(output_file_name, "w+") as fp:
        json.dump(valid_scenes, fp, indent='\t')
    print("[ INFO ] Valid scene named dumped.")


if __name__ == "__main__":
    
    parser = ArgumentParser("Argument parser for feed-forward training.")
    parser.add_argument("--config", type=str, default=os.path.join(os.getcwd(), "configs", "main.yaml"), 
                        help="Path to .yaml format config file.")
    args = parser.parse_args()
    cfg = LoadConfig(args.config)
    main(cfg)