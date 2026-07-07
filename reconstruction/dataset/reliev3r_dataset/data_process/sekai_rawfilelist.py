import os
from glob import glob
from tqdm import tqdm
import numpy as np



def main():
    # Scan all available files
    video_files_walking = glob(os.path.join("/efs-m3ga/h00548084/relive3r_raw/sekai/sekai-real-walking/1_clean", "*.mp4")) + \
                          glob(os.path.join("/efs-m3ga/h00548084/relive3r_raw/sekai/sekai-real-walking/3_scattered", "*.mp4")) + \
                          glob(os.path.join("/efs-m3ga/h00548084/relive3r_raw/sekai/sekai-real-walking/2_empty/VideoFrames", "*", "*.mp4")) + \
                          glob(os.path.join("/efs-m3ga/h00548084/relive3r_raw/sekai/sekai-real-walking/2_empty/CorpusExtract", "*", "*.mp4"))
    camera_files_walking = sorted(glob(os.path.join("/efs-m3ga/h00548084/relive3r_raw/sekai/sekai-real-walking-hq/*.npz")))
    video_files_walking_dict = {}
    camera_files_walking_dict = {}
    for video_file in video_files_walking:
        video_name = video_file.split("/")[-1].split('.')[0]
        video_files_walking_dict[video_name] = video_file
    for camera_file in camera_files_walking:
        camera_name = camera_file.split("/")[-1].split('.')[0]
        camera_files_walking_dict[camera_name] = camera_file
    print("[ INFO ] {} video files found.".format(len(video_files_walking_dict.keys())))
    print("[ INFO ] {} camera files found.".format(len(camera_files_walking_dict.keys())))
    
    # Match data pairs
    raw_data_pairs_unmatched = []
    raw_data_pairs_matched = []
    for video_name in tqdm(video_files_walking_dict.keys()):
        video_file = video_files_walking_dict[video_name]
        camera_file = camera_files_walking_dict.get(video_name, None)
        # os.system("sudo chmod 777 {}".format(video_file))
        # if camera_file is not None:
        #     os.system("sudo chmod 777 {}".format(camera_file))
        if camera_file is not None:
            raw_data_pairs_matched.append(dict(video=video_file, camera=camera_file, name=video_name))
        else:
            raw_data_pairs_unmatched.append(dict(video=video_file, camera=None, name=video_name))
    # raw_data_pairs = raw_data_pairs_unmatched + raw_data_pairs_matched
    raw_data_pairs = raw_data_pairs_matched
    print("[ INFO ] {} unmatched video-camera pairs found.".format(len(raw_data_pairs_unmatched)))
    print("[ INFO ] {} matched video-camera pairs found.".format(len(raw_data_pairs_matched)))
    np.savez("sekai_raw_data_pairs.npz", data=raw_data_pairs, allow_pickle=True)


if __name__ == '__main__':
    main()    