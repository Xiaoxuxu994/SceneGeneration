import torch
import torch.nn as nn

from utils.platform import DEVICE_NAME

class MatchingModel(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.name == "LightGlue":
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import rbd
            self.feat_extractor = SuperPoint(ckpt_path=cfg.ckpt_path.lightglue.superpoint, max_num_keypoints=None)
            self.feat_matcher   = LightGlue(ckpt_path=cfg.ckpt_path.lightglue.lightglue, features='superpoint')
        elif cfg.name == "CoTracker":
            from cotracker.predictor import CoTrackerPredictor
            self.tracker = CoTrackerPredictor(checkpoint=cfg.ckpt_path.cotracker, v2=False, offline=True, window_len=60)
        else:
            raise NotImplementedError(f"Matching model type {cfg.name} is not implemented.")
    
    @torch.no_grad()
    def infer(self, inputs):
        if self.cfg.name == "LightGlue":
            # Inputs are assumed as a set list of length B of paired tensors of shape (C, H, W)
            match_points = []
            for img0, img1 in inputs:
                assert img0.shape == img1.shape and img0.ndim == 3 and img0.shape[0] == 3
                feats0 = self.feat_extractor.extract(img0)
                feats1 = self.feat_extractor.extract(img1)
                matches01 = self.feat_matcher({"image0": feats0, "image1": feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                matches = matches01["matches"]
                points0 = feats0["keypoints"][matches[..., 0]]
                points1 = feats1["keypoints"][matches[..., 1]]
                match_points.append((points0, points1))
            # match_points is a list of length B of matching points pairs, 
            # each pair is composed of two tensors of shape (N, 2) respectively.
            return match_points
        elif self.cfg.name == "CoTracker":
            # Inputs are assumed as a batch of images, a tensor of shape (B, V, C, H, W)
            pred_tracks, pred_visibility = self.tracker(inputs, grid_size=50, grid_query_frame=0, backward_tracking=False)
            # pred_tracks: B V N 2, pred_visibility: B V N 1
            return pred_tracks, pred_visibility
        else:
            raise NotImplementedError(f"Matching model type {self.cfg.name} is not implemented.")
            