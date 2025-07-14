import torch
import numpy as np
import open3d as o3d

import os
import torch
import pandas as pd
import json
from scipy.spatial import KDTree
import math
import matplotlib.pyplot as plt
from skimage import measure
import cv2
import argparse
from tqdm import tqdm
"""  
    load udf surface
    random generate 50 camera poses 
    then generate render image
"""
def getR(ax,ay,az):
    # 绕 Z 轴的旋转矩阵（经度）
    R_z = np.array([
        [np.cos(ax), -np.sin(ax), 0],
        [np.sin(ax), np.cos(ax), 0],
        [0, 0, 1]
    ])
    # 绕 Y 轴的旋转矩阵（纬度）
    R_y = np.array([
        [np.cos(ay), 0, -np.sin(ay)],
        [0, 1, 0],
        [np.sin(ay), 0, np.cos(ay)]
    ])
    # 绕 X 轴的旋转矩阵（纬度）
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(az), -np.sin(az)],
        [0, np.sin(az), np.cos(az)]
    ])
    # 计算最终旋转矩阵（先纬度再经度）
    R = R_x @ R_y @ R_z  # 旋转顺序可能影响最终方向，可根据需要调整
    return R
def listobj(obj):
    for attr in dir(obj):
        if not attr.startswith("__"):  # 过滤掉特殊方法
            print(f"{attr}: {getattr(obj, attr)}")
GenMVI = True
# occ2udf2mcmesh.py 输出的mcmesh所在路径
mcmesh_path = 'udf/bboxzero_scannetpp_v1/udf_general'
# 多视角图像输出路径
outpath = 'udf/bboxzero_scannetpp_v1/mvr_general'
view_cfg = [4, 'False']
file_list = os.listdir(mcmesh_path)
grid_size = 0.05
alpha = 2
width = 960
height = 640
# 多视角图片生成的参数
num_list = [10,10,10,10,10,10]
yaw_list = [0,math.pi/6,math.pi/3,math.pi/6,math.pi/4,math.pi/3]
random_list = [True,True,True,True,True,True]
zoom_list = [0.6,0.6,0.6,0.15,0.25,0.35]
surfaceid_list = [0,0,0,1,1,1]
for _, file in tqdm(enumerate(file_list), desc='Processing'):      
    if not GenMVI:  # 显示交互界面 
        ## 得到两个曲面，一个是移除天花板，一个没有移除天花板
        tmp = np.load(os.path.join(mcmesh_path, file))
        prefix = file.rsplit('.')[0]
        udf = tmp['udf']
        # whole surface
        vertices = tmp['vert']
        faces = tmp['face']
        # surface after remove ceiling
        udf[:,:,(udf.shape[2]//5*4):udf.shape[2]] = np.inf
        vertices2, faces2, _, _ = measure.marching_cubes(udf, level=grid_size * alpha, spacing=(grid_size, grid_size, grid_size))
        
        ## 获得场景中心点，相机会朝向该点
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)
        center = (min_bound + max_bound) / 2
        
        ## 完整曲面场景  
        # open3d mesh setting
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.2, 0.4, 0.8])
        mesh.compute_vertex_normals()  # make sure light take effect
        # render window setting
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=not GenMVI, width=width, height=height) 
        # render setting
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True  
        render_option.light_on = True  # 确保光照启用
        render_option.background_color = [1, 1, 1]  # 亮背景减少对比
        vis.add_geometry(mesh)
        # 创建点云
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = mesh.vertices
        # pcd.colors = o3d.utility.Vector3dVector(np.full((len(vertices), 3), [1.0, 0.0, 0.0]))  # 红色突出显示        
        # vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        # 获取当前相机参数
        ctr.set_lookat(center)
        ctr.set_front(np.array([0,1,0]))
        ctr.set_up(np.array([0,0,1]))
        ctr.set_zoom(0.1)
        vis.run()
        vis.destroy_window()
    else: # 生成多视角图像
        tmp = np.load(os.path.join(mcmesh_path, file))
        prefix = file.rsplit('.')[0]
        udf = tmp['udf']
        angle = np.min([math.atan(udf.shape[0]/udf.shape[2]),math.atan(udf.shape[1]/udf.shape[2])])
        # whole surface
        vertices = tmp['vert']
        faces = tmp['face']
        # surface after remove ceiling
        udf[:,:,(udf.shape[2]//5*4):udf.shape[2]] = np.inf
        vertices2, faces2, _, _ = measure.marching_cubes(udf, level=grid_size * alpha, spacing=(grid_size, grid_size, grid_size))
        
        ## 获得场景中心点，相机会朝向该点
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)
        center = (min_bound + max_bound) / 2
        
        ## 完整曲面场景  
        # open3d mesh setting
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.2, 0.4, 0.8])
        mesh.compute_vertex_normals()  # make sure light take effect
        # render window setting
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=not GenMVI, width=width, height=height) 
        # render setting
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True  
        render_option.light_on = True  # 确保光照启用
        render_option.background_color = [1, 1, 1]  # 亮背景减少对比
        vis.add_geometry(mesh)

        ## 去除房顶的曲面场景  
        # open3d mesh setting
        mesh2 = o3d.geometry.TriangleMesh()
        mesh2.vertices = o3d.utility.Vector3dVector(vertices2)
        mesh2.triangles = o3d.utility.Vector3iVector(faces2)
        mesh2.paint_uniform_color([0.2, 0.4, 0.8])
        mesh2.compute_vertex_normals()  # make sure light take effect
        # render window setting
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(visible=not GenMVI, width=width, height=height) 
        # render setting
        render_option2 = vis2.get_render_option()
        render_option2.mesh_show_back_face = True  
        render_option2.light_on = True  # 确保光照启用
        render_option2.background_color = [1, 1, 1]  # 亮背景减少对比
        vis2.add_geometry(mesh2)

        vis_list = [vis, vis2]
        os.makedirs(os.path.join(outpath, prefix), exist_ok=True)
        os.makedirs(os.path.join(outpath, prefix, '0'), exist_ok=True)
        os.makedirs(os.path.join(outpath, prefix, '1'), exist_ok=True)
        cnt = 0   
        for item in zip(num_list, yaw_list, random_list, zoom_list, surfaceid_list):
            num, yaw, random, zoom, surfaceid = item
            if surfaceid == 1 and yaw < math.pi/2 - angle:
                yaw = math.pi/2 - angle 
            phi = np.linspace(0,math.pi*2,num+1)[0:-1]
            if random: 
                phi += np.random.uniform(-1,1,(1,))*math.pi

            for i in range(num): 
                p_i = phi[i]
                ctr = vis_list[surfaceid].get_view_control()
                ctr.set_lookat(center)
                ctr.set_front([-math.cos(yaw)*math.cos(p_i), math.sin(p_i)*math.cos(yaw), math.sin(yaw)]) 
                ctr.set_up(np.array([0,0,1]))
                ctr.set_zoom(zoom)

                vis_list[surfaceid].poll_events()
                vis_list[surfaceid].update_renderer()
                img = vis_list[surfaceid].capture_screen_float_buffer(do_render=True)
                img = (np.asarray(img) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(outpath, prefix, str(surfaceid), prefix+'_'+str(cnt + i)+'.png'), img)
            
            cnt += num
        vis_list[0].clear_geometries()
        vis_list[1].clear_geometries()
        vis_list[0].destroy_window()     
        vis_list[1].destroy_window()