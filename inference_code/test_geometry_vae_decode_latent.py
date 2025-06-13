import os
import torch
import torch_npu
from torch.utils.data import DataLoader
torch._C._jit_set_profiling_mode(False)
from tqdm import tqdm

## src code
from src.registry import MODELS, build_module
from src.models.geometryvae import *
from src.dataset.scannetppv2_dataset import ScanNetPPV2Dataset
from src.utils.config_utils import parse_configs
from src.utils.ckpt_utils import load_checkpoint
from src.utils.misc import save_inout_pc


def decode_and_save_latents(model, dataloader, latents, save_root, latent_file_name, device):
    for i in tqdm(range(latents.shape[0]), desc="Decoding latents"):
        latent = latents[i].unsqueeze(0).to(device)
        input_dict = next(iter(dataloader))  # Assumes consistent dataloader[0] usage
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key][0].to(device)

        with torch.no_grad():
            output = model.decode(latent, input_dict['pc_query'].unsqueeze(0))
            output_logit = output.squeeze(-1).cpu()

        scene_id = input_dict['scene_id'][0]
        save_dir = os.path.join(save_root, scene_id, f'{latent_file_name}_{i}')
        os.makedirs(save_dir, exist_ok=True)

        print(f"[INFO] Saving decoded output for scene: {scene_id}_{i}")
        save_inout_pc(save_dir, input_dict, output_logit)

def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs()
    print("=" * 30 + " cfg " + "=" * 30)
    for name in dir(cfg):
        if not name.startswith("__"):
            print(f"{name} = {getattr(cfg, name)}")
    print("=" * 60)
    device = torch.device('npu') #"cuda"
    dtype = torch.float32
    cfg.prefix = 'decode_latent'

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet
    dataset = ScanNetPPV2Dataset(
        data_root=cfg.data_root,
        data_list_name=cfg.val_data_list_name,
        aug_cfg = cfg.transform
    )

    dataloader = DataLoader(dataset,batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    # ======================================================
    # 3. exp output dir
    # ======================================================
    ckpt_path = os.path.join(cfg.root,cfg.ckpt_name,'model_ckpt.pt')
    exp_name = os.path.join(cfg.outputs, cfg.ckpt_name, cfg.prefix)
    os.makedirs(exp_name, mode = 0o777, exist_ok = True)
    print(f"Saving decoder latent to: {os.path.abspath(exp_name)}")

    # ======================================================
    # 4. build model
    # ======================================================
    print("====== model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint(model, ckpt_path, strict=True)
    model.eval()

    # =======================================================
    # 5. Decode latent Sample
    latent_file_root = '/home/z00533370/diffusion_result/data_latent/output_sample_pick_cpu/'
    latent_file_prefix = 'sl_flow_dit_XL_2048x512_fp16'
    latent_file = os.path.join(latent_file_root,latent_file_prefix,'step0020000/sample_step0020000.pth')
    latents = torch.load(latent_file)
    decode_and_save_latents(model, dataloader, latents, exp_name, latent_file_prefix, device)

if __name__ == "__main__":
    main()
    
    
    
