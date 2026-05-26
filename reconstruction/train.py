import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
from utils import PLATFORM, DEVICE_NAME

from argparse import ArgumentParser

import torch
if PLATFORM == "ASCEND":
    import torch_npu
    import torch_npu.npu as torch_device_lib
else:
    import torch.cuda as torch_device_lib
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from utils.config_loader import LoadConfig
from utils.image_utils import visualize_prediction
from utils.optim_utils import create_optimizer, create_lr_scheduler
from utils.log_utils import export_pcds_and_cam_as_ply
from utils import make_loss_func
from utils.common import normalize_trans, calc_extr_inv, localpoint2global
from dataset import Reliev3R_MixDataLoader

from model import Reliev3R, Reliev3R_Pi3


ENABLE_TIME_PROFILING = True

def clean_up():
    dist.destroy_process_group()


def create_dataloaders(rank, global_rank, cfg):
    """Create data loaders. """
    dataloader, dataloader_eval = None, None
    if not cfg.eval:
        dataloader = Reliev3R_MixDataLoader(cfg.datasets, "train", cfg.training.batch_size)
    if global_rank == 0:
        dataloader_eval = Reliev3R_MixDataLoader(cfg.datasets, "test", cfg.training.batch_size)
    dist.barrier()
    return dataloader, dataloader_eval


def dataloader_step(data_iter, dataloader, step):
    """Sample a data sample from dataloader. """
    epoch_ends = False
    while True:
        try:
            data = next(data_iter)
            """'data' returns None in case error is encountered in dataloader. """
            if data is not None:
                data = load_data_to_device(data)
                break
            print("[ WARNING ] Error data found at step {}!".format(step))
        except StopIteration:
            # if train_proc is not None:
            #     for ds_idx in range(len(dataloader.dataset.datasets)):
            #         dataloader.dataset.datasets[ds_idx].update_train_proc(train_proc)
            data_iter = iter(dataloader)
            epoch_ends = True
    return data, data_iter, epoch_ends


def export_image_samples(cfg, step, subfolder, data, pred, loss_dict):
    """Visualize images and depths. """
    os.makedirs(os.path.join(cfg.output_dir, subfolder), exist_ok=True)
    verbose_image = visualize_prediction(pred, data, loss_dict["matching_info"][-1])
    torchvision.utils.save_image(verbose_image, os.path.join(cfg.output_dir, subfolder, "iter_{:06d}.png").format(step))
    print("[ INFO ] Images saved successfully.")


def export_point_cloud(cfg, step, subfolder, data, pred, thre=0.5):
    """Visualize point maps. """
    os.makedirs(os.path.join(cfg.output_dir, subfolder), exist_ok=True)
    
    with torch.no_grad():
        print("[ INFO ] Visualizing with GT camera: {}".format(pred.get("camera_extrs", None) is None))
        cam_poses = data["input_viewmats"] if pred.get("camera_extrs", None) is None else pred["camera_extrs"]
        pred_points = localpoint2global(rearrange(pred["local_points"], "b v c h w -> b v h w c"), cam_poses)
        pred_points_conf = pred["depths_conf"]
    
    export_pcds_and_cam_as_ply(os.path.join(cfg.output_dir, subfolder, "step{:07d}".format(step)), 
                               pred_points, data["input_images"], cam_poses, 
                               pred_points_conf, conf_thre=thre)
    if data.get("input_points", None) is not None:
        export_pcds_and_cam_as_ply(os.path.join(cfg.output_dir, subfolder, "step{:07d}_GT".format(step)), 
                                   data["input_points"], data["input_images"], data["input_viewmats"], 
                                   data["input_depth_masks"], conf_thre=thre)
    print("[ INFO ] Point maps saved successfully.")


def compute_camera_metrics_and_log(pred, data, step, tb_writer, eval_angle, eval_trajectory, header):
    try:
        pr_cam = pred["camera_extrs"][0].detach()
        gt_cam = data["input_viewmats"][0]
        pr_cam_norm = normalize_trans(pr_cam)
        gt_cam_norm = normalize_trans(gt_cam)
        pr_cam_norm = pr_cam_norm[:1, ...] @ calc_extr_inv(pr_cam_norm)
        gt_cam_norm = gt_cam_norm[:1, ...] @ calc_extr_inv(gt_cam_norm)
        rra_acc, rta_acc, auc_at_thre = eval_angle(pr_cam_norm, gt_cam_norm)
        ate, rpe_trans, rpe_rot = eval_trajectory(pr_cam_norm, gt_cam_norm)
        tb_writer.add_scalar("{}_{}/cam_acc/rra_acc".format(header, data["dataset_names"][0]), rra_acc, step)
        tb_writer.add_scalar("{}_{}/cam_acc/rta_acc".format(header, data["dataset_names"][0]), rta_acc, step)
        tb_writer.add_scalar("{}_{}/cam_acc/auc_at_thre".format(header, data["dataset_names"][0]), auc_at_thre, step)
        tb_writer.add_scalar("{}_{}/cam_err/ate".format(header, data["dataset_names"][0]), ate, step)
        tb_writer.add_scalar("{}_{}/cam_err/rpe_trans".format(header, data["dataset_names"][0]), rpe_trans, step)
        tb_writer.add_scalar("{}_{}/cam_err/rpe_rot".format(header, data["dataset_names"][0]), rpe_rot, step)
    except:
        pass


def compute_depth_metrics_and_log(pred, data, step, tb_writter, eval_depth_map, header):
    try:
        gt_depth = data["input_depths"][0]            # v 1 h w
        gt_mask  = data["input_depth_masks"][0] > 0.0 # v 1 h w
        pr_depth = pred["depths"][0].detach()  # v 1 h w
        metrics = eval_depth_map(pr_depth, gt_depth, gt_mask, align_with_lstsq=True, device=DEVICE_NAME)
        tb_writter.add_scalar("{}_{}/depth/abs_rel".format(header, data["dataset_names"][0]), metrics["Abs Rel"], step)
        tb_writter.add_scalar("{}_{}/depth/δ<1.25".format(header, data["dataset_names"][0]), metrics["δ < 1.25"], step)
    except:
        pass


def run(rank, cfg):
    print("[ INFO ] Start DDP on rank {}".format(rank))
    
    torch_device_lib.set_device(rank)
    dist.init_process_group("nccl" if PLATFORM == "NVIDIA" else "hccl")
    # torch.autograd.detect_anomaly(True)
    
    """Get global rank for training output. """
    global_rank = rank + int(os.environ.get("VC_TASK_INDEX", 0)) * int(os.environ.get("MA_NUM_GPUS", 0))
    if global_rank == 0:
        from utils.eval_utils import eval_angle, eval_trajectory, eval_depth_map
    
    """Create data loaders. """
    dataloader, dataloader_eval = create_dataloaders(rank, global_rank, cfg)
    
    """Load models. """
    if cfg.get("train_pi3", False):
        model = Reliev3R_Pi3(cfg.model).to(device=DEVICE_NAME)
    else:
        model = Reliev3R(cfg.model).to(device=DEVICE_NAME)
    
    """Create loss functions. """
    loss_func = make_loss_func(cfg.loss)
    
    """Load model checkpoint. """
    step = 0
    if cfg.get("ckpt_path") is not None:
        print("[ INFO ] Loading checkpoint from local path: {}".format(cfg.ckpt_path))
        ckpt_info = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_info["model"])
        model = model.to(device=DEVICE_NAME)    
        del ckpt_info
        
        """Create optimizers based on model parameters in checkpoint. """
        optimizer = create_optimizer(model, 0.05, cfg.training.learning_rate)
        lr_scheduler = create_lr_scheduler(optimizer, cfg.training.iterations, cfg.training.warm_up_iterations, 
                                           scheduler_type=cfg.training.get("lr_scheduler", "cosine"))
        if not cfg.no_load_optim:
            ckpt_optim_info = torch.load(cfg.ckpt_path[:-5] + "_optim.ckpt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(ckpt_optim_info["optimizer"])
            lr_scheduler.load_state_dict(ckpt_optim_info["lr_scheduler"])
            step = ckpt_optim_info["current_iteration"]
            del ckpt_optim_info
            print("[ DEBUG ] Optimizer loading enabled.")
        else:
            print("[ DEBUG ] Optimizer loading disabled.")
        print("[ INFO ] Checkpoint loaded successfully")
    else:
        """Create optimizers from scratch. """
        optimizer = create_optimizer(model, 0.05, cfg.training.learning_rate)
        lr_scheduler = create_lr_scheduler(optimizer, cfg.training.iterations, cfg.training.warm_up_iterations, 
                                           scheduler_type=cfg.training.get("lr_scheduler", "cosine"))
    step_begins = step
    
    """Wrap model with ddp. """
    model = DDP(model, device_ids=[rank], bucket_cap_mb=100, gradient_as_bucket_view=True)
    
    model.train()
    torch_device_lib.empty_cache()
    dist.barrier()
    
    enable_time_profiling = ENABLE_TIME_PROFILING
    if enable_time_profiling:
        step_start,     step_end     = torch_device_lib.Event(enable_timing=True), torch_device_lib.Event(enable_timing=True)
        data_start,     data_end     = torch_device_lib.Event(enable_timing=True), torch_device_lib.Event(enable_timing=True)
        forward_start,  forward_end  = torch_device_lib.Event(enable_timing=True), torch_device_lib.Event(enable_timing=True)
        loss_start,     loss_end     = torch_device_lib.Event(enable_timing=True), torch_device_lib.Event(enable_timing=True)
        backward_start, backward_end = torch_device_lib.Event(enable_timing=True), torch_device_lib.Event(enable_timing=True)
        time_profile_table = {"step": 0.0, "data": 0.0, "forward": 0.0, "loss": 0.0, "backward": 0.0}
    
    TRAIN_PCD_VISUALIZE_THRE = 0.5
    EVAL_PCD_VISUALIZE_THRE  = 0.5
    
    if not cfg.eval:
        """Perform training. """
        if global_rank == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            """Create tensorboard writer. """
            tb_writer = SummaryWriter(log_dir=cfg.output_dir)
            """Save configuration file for reproduction. """
            with open(os.path.join(cfg.output_dir, "config.json"), "w+") as f:
                json.dump(cfg, f, indent='\t')
    
        """Training loop. """
        dataset_count = {}
        while step < cfg.training.iterations:
            
            if enable_time_profiling:
                step_start.record()
                data_start.record()
            
            data = dataloader.sample(step)
            step += 1
            if dataset_count.get(data["dataset_names"][0], None) is None:
                dataset_count[data["dataset_names"][0]] = 1
            else:
                dataset_count[data["dataset_names"][0]] += 1
            if step % 100 == 0 and rank == 0:
                print("[ INFO ] Step {}, dataset sample summarize:".format(step))
                print(dataset_count)
            
            if enable_time_profiling:
                data_end.record()
            
            """Training"""
            if enable_time_profiling:
                forward_start.record()
            
            model.train()
            with torch.autocast(device_type=DEVICE_NAME, dtype=torch.bfloat16):
                pred = model(data)

            if enable_time_profiling:
                forward_end.record()
                loss_start.record()

            loss_dict = loss_func(pred, data)
            
            if enable_time_profiling:
                loss_end.record()
                backward_start.record()
            
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(recurse=True), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            """Output training time profiling"""
            if enable_time_profiling and global_rank == 0:
                backward_end.record()
                step_end.record()
                torch_device_lib.synchronize()
                step_time     = step_start.elapsed_time(step_end) / 1000
                data_time     = data_start.elapsed_time(data_end) / 1000
                forward_time  = forward_start.elapsed_time(forward_end) / 1000
                loss_time     = loss_start.elapsed_time(loss_end) / 1000
                backward_time = backward_start.elapsed_time(backward_end) / 1000
                time_profile_table["step"]     += step_time
                time_profile_table["data"]     += data_time
                time_profile_table["forward"]  += forward_time
                time_profile_table["loss"]     += loss_time
                time_profile_table["backward"] += backward_time
                if step % 100 == 0 or step - step_begins <= 20:
                    print("======================================")
                    print("[ PROFILE - STEP ] step time:     {:.2f} seconds".format(step_time))
                    print("                   data time:     {:.2f} seconds, {:.2f}%".format(data_time, data_time / step_time * 100))
                    print("                   forward time:  {:.2f} seconds, {:.2f}%".format(forward_time, forward_time / step_time * 100))
                    print("                   loss time:     {:.2f} seconds, {:.2f}%".format(loss_time, loss_time / step_time * 100))
                    print("                   backward time: {:.2f} seconds, {:.2f}%".format(backward_time, backward_time / step_time * 100))
                    mean_step_time     = time_profile_table["step"] / (step - step_begins)
                    mean_data_time     = time_profile_table["data"] / (step - step_begins)
                    mean_forward_time  = time_profile_table["forward"] / (step - step_begins)
                    mean_loss_time     = time_profile_table["loss"] / (step - step_begins)
                    mean_backward_time = time_profile_table["backward"] / (step - step_begins)
                    print("[ PROFILE - MEAN ] step time:     {:.2f} seconds".format(mean_step_time))
                    print("                   data time:     {:.2f} seconds, {:.2f}%".format(mean_data_time, mean_data_time / mean_step_time * 100))
                    print("                   forward time:  {:.2f} seconds, {:.2f}%".format(mean_forward_time, mean_forward_time / mean_step_time * 100))
                    print("                   loss time:     {:.2f} seconds, {:.2f}%".format(mean_loss_time, mean_loss_time / mean_step_time * 100))
                    print("                   backward time: {:.2f} seconds, {:.2f}%".format(mean_backward_time, mean_backward_time / mean_step_time * 100))
                    print("======================================")
            
            if global_rank == 0:
                """Log training loss information"""
                print("[ INFO ] Step {}; Loss: {:.5f}; XY range: [{:.3f}, {:.3f}]; Z range: [{:.3f}, {:.3f}]; Conf range: [{:.3f}, {:.3f}]; Cam coord range: [{:.3f}, {:.3f}]; Dataset: {}".format(
                    step, loss_dict["total"].item(), 
                    pred["local_points"][..., :2, :, :].min().item(), pred["local_points"][..., :2, :, :].max().item(), 
                    pred["local_points"][..., -1:, :, :].min().item(), pred["local_points"][..., -1:, :, :].max().item(), 
                    pred["depths_conf"].min().item(), pred["depths_conf"].max().item(), 
                    pred["camera_extrs"][..., :3, 3].min().item(), pred["camera_extrs"][..., :3, 3].max().item(), 
                    data["dataset_names"][0]))
                for loss_name, loss_value in loss_dict.items():
                    if loss_name == "matching_info":
                        continue
                    tb_writer.add_scalar("loss/{}".format(loss_name), loss_value.item(), step)
                compute_camera_metrics_and_log(pred, data, step, tb_writer, eval_angle, eval_trajectory, "metric_train")
            
                if step % cfg.training.verbose_interval == 0:
                    """Visualize image, tracking, depth maps and confidence maps"""
                    export_image_samples(cfg, step, "vis_img_train", data, pred, loss_dict)
                    """Visualize point cloud"""
                    export_point_cloud(cfg, step, "vis_pcd_train", data, pred, thre=TRAIN_PCD_VISUALIZE_THRE)
                
                """Validation, only performed on global rank 0"""
                if step % cfg.training.eval_interval == 0 and global_rank == 0:
                    data = dataloader_eval.sample(step)
                    with torch.no_grad():
                        pred = model.module._forward(data)
                        loss_dict = loss_func(pred, data)
                        print("[ INFO - val ] Step {}; Loss: {:.5f}; XY range: [{:.3f}, {:.3f}]; Z range: [{:.3f}, {:.3f}]; Conf range: [{:.3f}, {:.3f}]; Cam coord range: [{:.3f}, {:.3f}]".format(
                            step, loss_dict["total"].item(), 
                            pred["local_points"][..., :2, :, :].min().item(), pred["local_points"][..., :2, :, :].max().item(), 
                            pred["local_points"][..., -1:, :, :].min().item(), pred["local_points"][..., -1:, :, :].max().item(), 
                            pred["depths_conf"].min().item(), pred["depths_conf"].max().item(), 
                            pred["camera_extrs"][..., :3, 3].min().item(), pred["camera_extrs"][..., :3, 3].max().item()))
                        for loss_name, loss_value in loss_dict.items():
                            if loss_name == "matching_info":
                                continue
                            tb_writer.add_scalar("loss_val/{}".format(loss_name), loss_value.item(), step)
                        """Report camera metrics of a single sample."""
                        compute_camera_metrics_and_log(pred, data, step, tb_writer, eval_angle, eval_trajectory, "metric_val")
                        compute_depth_metrics_and_log(pred, data, step, tb_writer, eval_depth_map, "metric_val")
                
                """Save model checkpoint. """
                if (step % cfg.training.ckpt_interval == 0 or step == cfg.training.iterations) and global_rank == 0:
                    print("[ INFO ] Saving checkpoint at iteration {}".format(step))
                    os.makedirs(os.path.join(cfg.output_dir, "ckpt"), exist_ok=True)
                    ## Save model weights
                    ckpt_name = os.path.join(cfg.output_dir, "ckpt", "{}.ckpt".format(step))
                    checkpoint = {
                        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), 
                    }
                    torch.save(checkpoint, ckpt_name)
                    ## Save model weights
                    ckpt_optim_name = os.path.join(cfg.output_dir, "ckpt", "{}_optim.ckpt".format(step))
                    checkpoint_optim = {
                        "optimizer": optimizer.state_dict(), 
                        "lr_scheduler": lr_scheduler.state_dict(), 
                        "current_iteration": step, 
                    }
                    torch.save(checkpoint_optim, ckpt_optim_name)
                    del checkpoint
                    del checkpoint_optim
    else:
        """Visualize evaluation results on eval set, not for metric computation, implemented for single GPU. """
        print("[ INFO ] Evaluation output directory: {}".format(cfg.output_dir))
        model.eval()
        eval_ends = False
        step_begins = step
        while True:
            data = dataloader_eval.sample(step)
            step += 1
            if eval_ends or step - step_begins > 100:
                break
            with torch.no_grad():
                pred = model.module._forward(data)
                loss_dict = loss_func(pred, data)
            """Visualize depth maps. """
            export_image_samples(cfg, step, "vis_img_eval", data, pred, loss_dict)
            """Export point cloud. """
            export_point_cloud(cfg, step, "vis_pcd_eval", data, pred, thre=EVAL_PCD_VISUALIZE_THRE)
            print("[ INFO ] Eval step {} done.".format(step - step_begins))
    clean_up()


def main(cfg):
    run(int(os.environ["LOCAL_RANK"]), cfg)


if __name__ == '__main__':
    parser = ArgumentParser("Argument parser for feed-forward training.")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "main.yaml"), 
                        help="Path to .yaml format config file.")
    parser.add_argument("--data_url", type=str, default=None, help="Input path for ModelArts training.")
    parser.add_argument("--train_url", type=str, default=None, help="Output path for ModelArts training.")
    parser.add_argument("--iterations", type=int, default=None, help="Training iterations, overwriting the configuration file.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size, overwriting the configuration file.")
    parser.add_argument("--ckpt_interval", type=int, default=None, help="Checkpoint save interval, overwriting the configuration file.")
    parser.add_argument("--verbose_interval", type=int, default=None, help="Prediction verbose interval, overwriting the configuration file.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--no_load_optim", action="store_true", help="Set to disable loading optimizer when ckpt is given.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Maximum learning rate.")
    parser.add_argument("--eval", action="store_true", help="Evaluate only.")
    args = parser.parse_args()
    
    cfg = LoadConfig(args.config)
    
    if args.data_url is not None:
        cfg.datasets.dl3dv.data_root = args.data_url
    if args.train_url is not None:
        cfg.output_dir = args.train_url
    if args.iterations is not None:
        cfg.training.iterations = args.iterations
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.ckpt_interval is not None:
        cfg.training.ckpt_interval = args.ckpt_interval
    if args.verbose_interval is not None:
        cfg.training.verbose_interval = args.verbose_interval
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    cfg.eval = args.eval
    cfg.no_load_optim = args.no_load_optim
    
    main(cfg)
