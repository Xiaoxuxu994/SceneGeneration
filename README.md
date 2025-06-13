# SceneGeneration

空间大模型--3D场景生成模型

代码分支：
- 3dgvae\_npu\_ljr: 3D Geometry VAE NPU版本 by ljr。



**3DGeometryVAE Train Code NPU 代码**
**2025.06.13**
新增fps_cpu模块，用cpu算子，基于simd和kd-tree开发的一个适用于小规模fps采样（$N<=120k$）的算子。安装步骤如下：
1. 从源代码安装nanoflann:
'''
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann
mkdir build && cd build
cmake ..
make
sudo make install
'''

2. 按如下命令安装fps_cpu:
'''
cd ops
python setup.py install
cd ..
'''

3.在config中开启 use_cpu_fps=True:

'''
model = dict(
    ...,
    decoder_dict=dict(
        ...
    ),
    use_cpu_fps=True # 推荐添加在这里
)

batch_size = 1
...

即可运行fps_cpu采样。910B DDP开启时，端到端每次前+反向时延应<2.3s。

**2025.06.03**
单卡运行命令： `ASCEND_RT_VISIBLE_DEVICES=0 python train_npu_single.py --config configs/exp0417_3D_geometry_vae_releaseV1.py`
0603版本复现（训练）精度： mIoU 66.42 Precision 82.26 Recall 77.14 F1 79.30 (161Epoch)

多卡运行命令：
'''bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_npu_col.py \
    --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py
'''
适配方案：git clone开源Colossalai，仅使用其通信域定义和dataloader sampler包装，模型使用DDP进行包装和存储，暂不进行FSDP。
dataset: `/mnt/nvme3/z00533370/dataset/scannetpp_trainval_v6`


**3DGeometryVAE Train Code GPU 代码**
**2025.05.29**
运行命令： `CUDA_VISIBLE_DEVICES=0,1,2,3 colossalai run --nproc_per_node 4 --mater_port 20953 train_gpu.py --config configs/exp0417_3D_geometry_vae_releaseV1.py`
复现精度： mIoU 79.30 Precision 85.92 Recall 90.83 F1 88.14
dataset: `/mnt/nvme3/z00533370/dataset/scannetpp_trainval_v6`

**3DGeometryVAE Decoder GPU 代码**
**2025.05.14**
1. ckpt: 包含两个模型的ckpt文件:
    `exp0417_VecSetAE_PTV3_NetArc_V6-2_4gpu_voxel0p05_decoder8_datav2_latent2048`
    `exp0506_VecSetAE_PTV3_NetArc_V6-2_4gpu_voxel0p05_decoder8_datav2_latent2048_updataset3`

2. inference_code: 推理代码，包含两个模型的config：
    releaseV1版本模型，对应上一版本dump出来的latent：`configs/exp0417_3D_geometry_vae_releaseV1.py `
    releaseV2版本模型，核心不同点是重构了dataloader：`configs/exp0506_3D_geometry_vae_releaseV2.py`

3. 推理命令：
    - releaseV1后缀对应releaseV1模型：
       ***eval mode***
       ` CUDA_VISIBLE_DEVICES=0 python test_geometry_vae_releasev1.py --config configs/exp0417_3D_geometry_vae_releaseV1.py `

       ***dump encode latent***
        `CUDA_VISIBLE_DEVICES=0 python test_geometry_vae_dump_latent_releasev1.py --config configs/exp0417_3D_geometry_vae_releaseV1.py `

       ***decode***
        对diffusion生成latent进行解码：`CUDA_VISIBLE_DEVICES=0 python test_geometry_vae_decode_latent_releasev1.py --config configs/exp0417_3D_geometry_vae_releaseV1.py `
        基于encoder dump的latent文件进行精度评估：`CUDA_VISIBLE_DEVICES=0 python test_geometry_vae_decode_latent_eval_releasev1.py --config configs/exp0417_3D_geometry_vae_releaseV1.py `
    
    - releaseV2后缀对应releaseV2模型：
       ***eval mode***
        `CUDA_VISIBLE_DEVICES=0 python test_geometry_vae.py --config configs/exp0506_3D_geometry_vae_releaseV2.py`

       ***dump encode latent***
        `CUDA_VISIBLE_DEVICES=0 python test_geometry_vae_dump_latent.py --config configs/exp0506_3D_geometry_vae_releaseV2.py`

       ***decode***
        对diffusion生成latent进行解码：`CUDA_VISIBLE_DEVICES=0 python test_geometry_vae_decode_latent.py --config configs/exp0506_3D_geometry_vae_releaseV2.py`
        基于encoder dump的latent文件进行精度评估：`CUDA_VISIBLE_DEVICES=0 python ttest_geometry_vae_decode_latent_eval.py --config configs/exp0506_3D_geometry_vae_releaseV2.py`

4. 模型精度 scannet++V2 Val集
    - releaseV1：mIoU 79.30 Precision 85.92 Recall 90.83 F1 88.14
    - releaseV2: mIoU 81.72 Precision 88.25 Recall 91.60 F1 89.86

**pre-process后训练数据路径**
    `/home/y00391175/project_xiaoxu/dataset/3DGS/scannetpp_trainval_v6/`

**Dump latent 数据路径**
    - releaseV1：`/home/data/proj_xiaoxu/scannetpp_latent_2048_512/`
    - releaseV2：`/home/data/proj_xiaoxu/scannetpp_latent_2048_512_v2/`