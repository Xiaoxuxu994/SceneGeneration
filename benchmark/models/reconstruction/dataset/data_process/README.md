# Data Protocals
Manuscript for Reliev3R data protocals. 

## TODO List
- [.] The most easy dataloader: Given a fixed image idxs, we can successfully load all data in batches and returns a data sample. 
- [.] Create dataloader for all datasets. 
- [.] Transform all extrinsics into world-to-camera format.
- [.] Integrate image resize. 
- [.] Integrate image sampling. 
- [.] Filter illegal values of of camera parameters.
- [.] Adapt view sampler to small clips.
- [.] Train/val split
- [.] Balance data sample ratio.
- [.] Adapt to batch size > 1, problem is to adapt different image resolutions from datasets. 
- [ ] Integrate image augmentation. 
- [ ] Depth mask check for each dataset.

## Data File Structure

### Realistic Data
We have datasets preprocessed and reorganized in file structure, which is expected as, 
```
|- dataset1 -|
|            |- sample1 -|
|            |           |- rgb -|
|            |           |       |- frame_000000.jpg
|            |           |       |- frame_000001.jpg
|            |           |       |- ...
|            |           |- moge_depth -|
|            |           |              |- frame_000000.exr
|            |           |              |- frame_000001.exr
|            |           |              |- ...
|            |           |- moge_depth_mask -|
|            |           |                   |- frame_000000.png
|            |           |                   |- frame_000001.png
|            |           |                   |- ...
|            |           |- cotracker3 -|
|            |           |              |- xxx.npz
|            |           |- sample1.npy
|            |- sample2
|            |- sample3
|            |- ...
|- dataset2
|- dataset3
|- ...
```

### Synthetic Data
Synthetic datasets remains the same with their raw file structure. 
GT depth maps are directly used, and matching information should be calculated online by reprojection. 

## Dataset Structure
Sample from Dataset class: 
```
|- images: (B,V,3,H,W)
|- depths: (B,V,1,H,W)
|- gt_points: (B,V,3,H,W), optional, to adapt with PI3 training.
|- cam_extrs: (B,V,4,4), world-to-camera
|- cam_intrs: (B,V,3,3), regularized
|- matchings: ((B,C(V,2),K,2), (B,C(V,2),K,2))
|- matching_masks: ((B,C(V,2),K), (B,C(V,2),K)) | (None, None)
```

## Data Utilities to be Implemented
* Image resizing into shapes adaptive to DINO transformer. 
* Image sampling strategy, 
    * Realistic dataset: Subsample from the splitted image clip. 
    * Synthetic dataset: Directly sample from the scene, may refer to VGGT-like dataloader. 
* Image augmentation, including, 
    * Color jitter, bluring, etc..
    * Image rotation (90 degrees), along with camera rotation along z-axis.
