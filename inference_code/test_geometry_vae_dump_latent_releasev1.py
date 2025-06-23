import os
import torch
from torch.utils.data import DataLoader
torch._C._jit_set_profiling_mode(False)
from tqdm import tqdm

## src code
from src.registry import MODELS, build_module
from src.models.geometryvae import *
from src.dataset.dataset_build_scannet_v2 import DatasetForScanNetV2Test
from src.utils.config_utils import parse_configs
from src.utils.ckpt_utils import load_checkpoint


def process_and_save_latent(model, input_dict, save_dir):
    encoder_latent, sample_pc = model.forward_encode_latent(input_dict)
    scene_id = input_dict['scene_id'][0]
    out_path = os.path.join(save_dir, f'{scene_id}.pth')
    torch.save({
        'latent': encoder_latent.cpu().squeeze(0),
        'sample_pc': sample_pc.cpu().squeeze(0)
    }, out_path)

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
    device = torch.device('npu:0')
    dtype = torch.float32
    cfg.prefix = 'embed_latent/val'

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet
    dataset = DatasetForScanNetV2Test(
        data_root=cfg.data_root,
        data_list_name=cfg.val_data_list_name,
        query_sample_rate=1.0,
        max_query_number=1000000,
        normalize_coord = cfg.normalize_coord,
        Gridsample_mode = 'train',
        shufflepoint = False,
        grid_size = cfg.grid_size
    )

    dataloader = DataLoader(dataset,batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    # ======================================================
    # 3. exp output dir
    # ======================================================
    ckpt_path = os.path.join(cfg.root,cfg.ckpt_name,'ckpt484.pth')
    exp_name = os.path.join(cfg.outputs, cfg.ckpt_name, cfg.prefix)
    os.makedirs(exp_name, mode = 0o777, exist_ok = True)
    print(f"Saving latent embeddings to: {os.path.abspath(exp_name)}")

    # ======================================================
    # 4. build model
    # ======================================================
    print("====== model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint(model, ckpt_path, strict=True)
    model.eval()
    profiling = False
    if profiling:
        import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
        prof = torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU,
                                    torch_npu.profiler.ProfilerActivity.CPU],
                        with_stack=True,
                        record_shapes=False,
                        profile_memory=True,
                        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=0,
                                                                skip_first=0),
                        experimental_config=experimental_config,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./profiling/v1_encode/'))
        prof.start()

    # =======================================================
    # 5. Dump Encode latent
    for num_pic, input_dict in tqdm(enumerate(dataloader), desc="Encoding latent"):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key][0].to(device)

        with torch.no_grad():
            process_and_save_latent(model, input_dict, exp_name)
            if profiling:
                prof.step()
                if num_pic > 5:
                    print('Stopping profiling.')
                    prof.stop()
                    exit(-1)

if __name__ == "__main__":
    main()
    
    
    
