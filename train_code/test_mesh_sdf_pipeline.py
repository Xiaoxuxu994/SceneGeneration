# %%
import os
import torch
from torch.utils.data import DataLoader
torch._C._jit_set_profiling_mode(False)
from tqdm import tqdm

## opensora 
from opensora_local.utils.ckpt_utils import  load_checkpoint
from opensora_local.utils.config_utils import parse_configs

## src code
from src.registry import MODELS, build_module,build_dataset_kwargs
from src.dataset.dataset_build_mesh_test import DatasetForScanNetV6TESTSDF
from src.utils.misc import save_inout_pc_input,save_surface_as_mesh
from src.utils.metric import extract_surface_points

if __name__ == '__main__':

    ## initialize model
    cfg = parse_configs(training=False)
    print("====== cfg: ",cfg)
    device = "npu:0"
    dtype = torch.float32

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet    
    val_dataset = DatasetForScanNetV6TESTSDF(
        data_root=cfg.data_root,
        data_list_name=cfg.val_data_list_name,
        **build_dataset_kwargs(cfg.test_data_args)
    )

    dataloader = DataLoader(val_dataset,batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    # ======================================================
    # 3. exp output dir
    # ======================================================
    cfg.outputs = '/mnt/nvme3/z00533370/test_result/'
    prefix = 'test_mode'
    ckpt_path = os.path.join(cfg.root,cfg.ckpt_name)
    exp_name = os.path.join(cfg.outputs, prefix)
    os.makedirs(exp_name, mode = 0o777, exist_ok = True)
    print("====== data saving to : ", exp_name)

    # ======================================================
    # 4. build model
    # ======================================================
    print("====== cfg.model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint(model, ckpt_path, save_as_pt=cfg.save_as_pt)
    model.eval()

    # =======================================================
    # 5. Test Loop
    # =======================================================
    scale_level = cfg.vs_res
    vs_c = scale_level[0]
    vs_f = scale_level[-1]
    for input_dict in tqdm(dataloader):
        mask_list = []
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key][0].to(device)

        with torch.no_grad():
            output_list = model.forward_test(input_dict,vs=scale_level[-1])

        # === save and print ===
        sub_folder = input_dict['scene_id'][0]
        inout_dir = os.path.join(exp_name, sub_folder)
        os.makedirs(inout_dir, exist_ok=True)
        save_inout_pc_input(inout_dir, input_dict)

        for idx,logit in enumerate(output_list):
            scale = scale_level[idx]
            grid_size = [scale,scale,scale]

            print(f"\n***************[vs resolution: {scale}]***************")
            pred_np_field = logit.view(*grid_size).detach().cpu().numpy()
            verts_pred, faces_pred = extract_surface_points(pred_np_field, voxel_size=2.0/scale, alpha = cfg.mc_alpha)
            if inout_dir is not None:
                mesh_path = f'{inout_dir}/pred_mesh_{scale}.ply'
                print(f"[INFO] Saving mesh to: {mesh_path}") 
                save_surface_as_mesh(mesh_path, verts_pred, faces_pred)
