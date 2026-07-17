import os
import time
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import argparse

SAMPLE_INTERVAL = 10
FRAME_PER_CLIP = 32
OUT_RESOLUTION_H = 288
OUT_RESOLUTION_W = 512

def main(args):
    raw_data_pairs_file = "./sekai_raw_data_pairs.npz"
    output_dir = "/efs-m3ga-hpc/c50041401/datasets/sekai/sekai_clips_all"
    raw_data_pairs = np.load(raw_data_pairs_file, allow_pickle=True)['data']
    # raw_data_pairs = raw_data_pairs[-32:]
    raw_data_pairs = raw_data_pairs[args.proc_id::args.total_proc]
    print("[ INFO - PROC {} ] {} data pairs found to process.".format(args.proc_id, len(raw_data_pairs)))
    # for raw_data_pair in tqdm(raw_data_pairs):
    total_time = 0.0
    processed_cnt = 0
    for raw_data_pair in raw_data_pairs:
        start_time = time.time()
        video_file = raw_data_pair['video']
        camera_file = raw_data_pair['camera']
        scene_name = raw_data_pair['name']
        
        ### Read video frames
        all_frames = []
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("[ WARNING ] Opening video '{}' failed.".format(video_file))
            continue
        # print("[ INFO - PROC {} ] Loading video frames...".format(args.proc_id))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (OUT_RESOLUTION_W, OUT_RESOLUTION_H))
            all_frames.append(frame)
        cap.release()
        
        ### Read camera information
        camera_info = np.load(camera_file)
        extrinsics = camera_info['extrinsic'] # N 4 4, opencv c2w
        intrinsics = camera_info['intrinsic'] # 1 3 3
        extrinsics = np.linalg.inv(extrinsics)
        
        ### Process images
        # for offset in tqdm(range(SAMPLE_INTERVAL), desc="Processing different offsets..."):
        for offset in range(SAMPLE_INTERVAL):
            clip_count = 0
            clip_frames = []
            clip_extrs = []
            clip_intrs = []
            for frame_idx, frame in enumerate(all_frames[offset::SAMPLE_INTERVAL]):
                clip_frames.append(frame)
                clip_extrs.append(extrinsics[offset + frame_idx * SAMPLE_INTERVAL])
                clip_intrs.append(intrinsics)
                if len(clip_extrs) >= FRAME_PER_CLIP:
                    clip_output_path = os.path.join(output_dir, "sub_{:06d}".format(offset), scene_name + "_{:03d}".format(clip_count))
                    clip_output_img_path = os.path.join(clip_output_path, "rgb")
                    clip_camera_path = os.path.join(clip_output_path, scene_name + "_{:03d}".format(offset) + ".npy")
                    os.makedirs(clip_output_path, exist_ok=True)
                    os.makedirs(clip_output_img_path, exist_ok=True)
                    for clip_frame_idx, clip_frame in enumerate(clip_frames):
                        cv2.imwrite(os.path.join(clip_output_img_path, "frame_{:05d}.jpg".format(clip_frame_idx)), clip_frame)
                    clip_extrs = np.stack(clip_extrs, axis=0)
                    clip_intrs = np.stack(clip_intrs, axis=0)
                    np.save(clip_camera_path, dict(extr_mat=clip_extrs, intr_mat=clip_intrs), allow_pickle=True)
                    
                    del clip_frames
                    del clip_extrs
                    del clip_intrs
                    clip_frames = []
                    clip_extrs = []
                    clip_intrs = []
                    clip_count += 1
        end_time = time.time()
        total_time += (end_time - start_time) / 3600 # in hours
        processed_cnt += 1
        print("[ INFO - PROC {} ] {}/{}={:.2f}% processed, {:.2f} min/scene, {:.2f} hours left".format(
            args.proc_id, 
            processed_cnt, len(raw_data_pairs), processed_cnt / len(raw_data_pairs) * 100, 
            total_time / processed_cnt * 60, 
            total_time / processed_cnt * (len(raw_data_pairs) - processed_cnt), 
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--proc_id", type=int, required=True)
    parser.add_argument("--total_proc", type=int, required=True)
    args = parser.parse_args()
    main(args)
