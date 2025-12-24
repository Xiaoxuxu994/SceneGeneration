import os
import json
import copy
import argparse
import pandas as pd
from easydict import EasyDict as edict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


import zipfile
import zlib
def check_integrity(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.testzip()  # 试图读取并验证压缩文件中的内容
        return True
    except zipfile.BadZipFile:
        print(f"[Warning] The file is corrupted or not a valid zip file. filepath is {file_path}")
        return False
    except zlib.error as e:
        # 捕获 zlib 解压缩错误，输出详细的错误信息
        print(f"[Warning] Zlib decompression error in file {file_path}. Error: {e}")
        return False
    except Exception as e:
        # 捕获其他可能的错误
        print(f"[Warning] An unexpected error occurred while checking file {file_path}. Error: {e}")
        return False
    

def process_sha256(sha256, input_dir, img_dir, opt, output_dir, progress_bar):
    try:
        # 加载数据
        ori_viewinfo = np.load(os.path.join(input_dir, f'{sha256}_viewInfo.npz'))
        ori_image = np.load(os.path.join(img_dir, f'{sha256}_img_feats_oriImgs.npy')) / 255
        ori_imgname = np.load(os.path.join(img_dir, f'{sha256}_img_feats_filenames.npy'))
        
        # 转换为字典
        viewinfo = dict(ori_viewinfo.items())
        
        # 读取相机信息
        cam_infos = json.load(open(os.path.join('/efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole', 'renders', sha256, 'transforms.json')))
        frames = cam_infos['frames']
        post_fix = frames[0]["file_path"].split('.')[-1]
        
        # 修改 img_filename
        new_imgname = []
        for imgname in ori_imgname:
            if len(imgname.split('.')) == 1:
                new_imgname.append(imgname + '.' + post_fix)
            elif len(imgname.split('.')) == 2:
                new_imgname.append(imgname)
            else:
                print(f'[ERROR] undefined image name : {imgname}')
        
        viewinfo['img_filename'] = new_imgname
        viewinfo['ori_image'] = ori_image
        
        # 打印图像的一些信息
        # print(f"ori_img: {ori_image.shape, ori_image.max(), ori_image.min()}")
        
        # 保存新的 viewinfo 文件
        np.savez_compressed(os.path.join(output_dir, f'{sha256}_viewInfo.npz'), **viewinfo)
        print(f"Saved new viewInfo: {viewinfo.keys()} to path: {os.path.join(output_dir, f'{sha256}_viewInfo.npz')}")
    
    except Exception as e:
        print(f"[ERROR] Failed to process {sha256}: {e}")
    
    # 更新进度条
    progress_bar.update(1)

def process_all_sha256s(sha256s, input_dir, img_dir, opt, output_dir, num_threads=4):
    # 创建进度条
    with tqdm(total=len(sha256s), desc="Processing sha256s", ncols=100) as progress_bar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交每个任务到线程池
            futures = []
            for sha256 in sha256s:
                futures.append(executor.submit(process_sha256, sha256, input_dir, img_dir, opt, output_dir, progress_bar))
            
            # 等待所有任务完成
            for future in futures:
                future.result()  

# === 主程序 ===
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole', 
                        help='Directory to save the metadata')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='start_index of handling instances')
    parser.add_argument('--end_idx', type=int, default=910,
                        help='start_index of handling instances')  
    opt = parser.parse_args()
    opt = edict(vars(opt))
    
    ori_dataset_dir = '/efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole'
    # '/efs-mi-east4-1/z84301856/datasets/scannetpp_v2_opencv_pinhole'
    
    input_dir = os.path.join(ori_dataset_dir, 'fix32_viewInfos')
    img_dir = os.path.join(ori_dataset_dir, 'features_norm128_v14_fix32', 'dinov2_vitl14_reg')
    output_dir = os.path.join(opt.dataset_dir, 'supervise_view_fix32_viewInfo_normV14')
    os.makedirs(output_dir, exist_ok=True)


    metadata_path = os.path.join(opt.dataset_dir, 'metadata_0.05.csv')
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[opt.start_idx : opt.end_idx]
    # metadata = metadata[ : 1]
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    # if '2748de13fb' in sha256s:
    #     sha256s.remove('2748de13fb')
    
    sha256s = ['c49a8c6cff']
    
    # for sha256 in copy.copy(sha256s):
    #     feat_file = os.path.join(output_dir,  f'{sha256}_viewInfo.npz') # f'{sha256}.npz')
    #     cond = os.path.exists(feat_file) and check_integrity(feat_file) 
    #     if cond :
    #         sha256s.remove(sha256)
    
    print(f"[INFO] Launching parallel viewInfo organization on {len(sha256s)} scenes...")       
    
    
    # 设置线程池大小，num_threads 可以根据你的机器性能进行调整
    process_all_sha256s(sha256s, input_dir, img_dir, opt, output_dir, num_threads=32)
    
    # for sha256 in tqdm(sha256s):
    #     ori_viewinfo = np.load(os.path.join(input_dir, f'{sha256}_viewInfo.npz'))
    #     ori_image = np.load(os.path.join(img_dir, f'{sha256}_img_feats_oriImgs.npy')) / 255
    #     ori_imgname = np.load(os.path.join(img_dir, f'{sha256}_img_feats_filenames.npy'))
        
    #     viewinfo = dict(ori_viewinfo.items())
        
    #     cam_infos = json.load(open(os.path.join(opt.dataset_dir, 'renders', sha256, 'transforms.json')))
    #     frames = cam_infos['frames']
    #     post_fix = frames[0]["file_path"].split('.')[-1]
    #     new_imgname = []
    #     for imgname in ori_imgname:
    #         if len(imgname.split('.')) == 1:
    #             new_imgname.append(imgname+'.'+post_fix)
    #         elif len(imgname.split('.')) == 2:
    #             new_imgname.append(imgname)
    #         else:
    #             print(f'[ERROR] undefined image name : {imgname}')
        
    #     viewinfo['img_filename'] = new_imgname
    #     viewinfo['ori_image'] = ori_image
    #     print(f"ori_img: {ori_image.shape, ori_image.max(), ori_image.min()}")        
    #     np.savez_compressed(os.path.join(output_dir, f'{sha256}_viewInfo.npz'), **viewinfo)
    #     print(f"save new viewInfo: {viewinfo.keys()} to path: {os.path.join(output_dir, f'{sha256}_viewInfo.npz')}")
        
        


