import warnings
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Tuple, Type, Union
import copy

from mapanything.models.mapanything.model import MapAnything, MultiViewTransformerInput
from uniception.models.encoders import ViTEncoderInput, ViTEncoderNonImageInput, EncoderGlobalRepInput
from mapanything.utils.geometry import normalize_pose_translations


class MapAnythingEncoder(MapAnything):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # self.cam_trans_scale_encoder_trained = copy.deepcopy(self.cam_trans_scale_encoder)
        # self.cam_trans_encoder_trained = copy.deepcopy(self.cam_trans_encoder)
        # self.cam_rot_encoder_trained = copy.deepcopy(self.cam_rot_encoder)
        # self.ray_dirs_encoder_trained = copy.deepcopy(self.ray_dirs_encoder)
    
    
    def _compute_pose_quats_and_trans_for_across_views_in_ref_view(
        self,
        views,
        num_views,
        device,
        dtype,
        batch_size_per_view,
        per_sample_cam_input_mask,
    ):
        """
        Gather pose quats and translations for all views, assuming
        they are already expressed in the reference cam0 frame.

        Returns identity pose for views where the camera input mask is False
        or the pose is not provided.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            device (torch.device): Device to use for the computation.
            dtype (torch.dtype): Data type to use for the computation.
            batch_size_per_view (int): Number of samples per view in the batch.
            per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

        Returns:
            torch.Tensor: A tensor containing the pose quats for all views in cam0 frame. (batch_size_per_view * num_views, 4)
            torch.Tensor: A tensor containing the pose trans for all views in cam0 frame. (batch_size_per_view * num_views, 3)
            torch.Tensor: The updated per sample camera input mask.
        """
        # Initialize as identity
        pose_quats_across_views = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device
        ).repeat(batch_size_per_view * num_views, 1)  # (q_x, q_y, q_z, q_w)
        pose_trans_across_views = torch.zeros(
            (batch_size_per_view * num_views, 3), dtype=dtype, device=device
        )

        for view_idx in range(num_views):
            per_sample_mask = per_sample_cam_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ]

            if (
                "camera_pose_quats" in views[view_idx]
                and "camera_pose_trans" in views[view_idx]
                and per_sample_mask.any()
            ):
                # Copy directly, since poses are already in cam0 frame
                pose_quats_across_views[view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view][per_sample_mask] = \
                    views[view_idx]["camera_pose_quats"][per_sample_mask].to(dtype=dtype)

                pose_trans_across_views[view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view][per_sample_mask] = \
                    views[view_idx]["camera_pose_trans"][per_sample_mask].to(dtype=dtype)
            else:
                # Mask out invalid samples
                per_sample_cam_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
                ] = False

        return pose_quats_across_views, pose_trans_across_views, per_sample_cam_input_mask
    
    
    def _encode_n_views(self, num_views, images, data_norm_type='dinov2'):
        """
        Encode all the input views (batch of images) in a single forward pass.
        Assumes all the input views have the same image shape, batch size, and data normalization type.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        # Reshape inputs flattening batch_size_per_view and num_views
        batch_size_per_view, num_views, _, _, _ = images.shape
        images = images.reshape(batch_size_per_view*num_views, *images.shape[2:]).contiguous()
        
        encoder_input = ViTEncoderInput(
            image=images, data_norm_type=data_norm_type
        )
        encoder_output = self.encoder(encoder_input)
        all_encoder_features_across_views = encoder_output.features.chunk(
            num_views, dim=0
        )
        
        return all_encoder_features_across_views
    
    
    def _encode_and_fuse_ray_dirs(
        self,
        ray_dirs,
        all_encoder_features_across_views,
    ):
        """
        Encode the ray directions for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            ray_dirs (torch.Tensor)
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Encode the ray directions
        ray_dirs_features_across_views = self.ray_dirs_encoder(
            ViTEncoderNonImageInput(data=ray_dirs)
        ).features

        # Sum to image features
        all_encoder_features_across_views = (
            all_encoder_features_across_views + ray_dirs_features_across_views
        )

        return all_encoder_features_across_views
    
    
    def _encode_and_fuse_cam_quats_and_trans(
        self,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        pose_quats_across_views,
        pose_trans_across_views,
        is_metric_scale
    ):
        """
        Encode the camera quats and trans for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            pose_quats_across_views (torch.Tensor): Tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
            pose_trans_across_views (torch.Tensor): Tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """        
        # Encode the pose quats
        pose_quats_features_across_views = self.cam_rot_encoder(
            EncoderGlobalRepInput(data=pose_quats_across_views)
        ).features

        # Get the metric scale mask for all samples
        device = all_encoder_features_across_views.device
        metric_scale_pose_trans_mask = torch.zeros(
            (batch_size_per_view * num_views), dtype=torch.bool, device=device
        )
        for view_idx in range(num_views):
            metric_scale_mask = is_metric_scale[view_idx]
            metric_scale_pose_trans_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ] = metric_scale_mask

        # Get the scale norm factor for all the samples and scale the pose translations
        pose_trans_across_views = torch.split(
            pose_trans_across_views, batch_size_per_view, dim=0
        )  # Split into num_views chunks
        pose_trans_across_views = torch.stack(
            pose_trans_across_views, dim=1
        )  # Stack the views along a new dimension (batch_size_per_view, num_views, 3)
        scaled_pose_trans_across_views, pose_trans_norm_factors = (
            normalize_pose_translations(
                pose_trans_across_views, return_norm_factor=True
            )
        )

        # Resize the pose translation back to (batch_size_per_view * num_views, 3) and extend the norm factor to (batch_size_per_view * num_views, 1)
        scaled_pose_trans_across_views = scaled_pose_trans_across_views.unbind(
            dim=1
        )  # Convert back to list of views, where each view has batch_size_per_view tensor
        scaled_pose_trans_across_views = torch.cat(
            scaled_pose_trans_across_views, dim=0
        )  # Concatenate back to (batch_size_per_view * num_views, 3)
        pose_trans_norm_factors_across_views = pose_trans_norm_factors.unsqueeze(
            -1
        ).repeat(num_views, 1)  # (B, ) -> (B * V, 1)

        # Encode the pose trans
        pose_trans_features_across_views = self.cam_trans_encoder(
            EncoderGlobalRepInput(data=scaled_pose_trans_across_views)
        ).features

        # Encode the pose translation norm factors using the log scale encoder for pose trans
        log_pose_trans_norm_factors_across_views = torch.log(
            pose_trans_norm_factors_across_views + 1e-8
        )
        pose_trans_scale_features_across_views = self.cam_trans_scale_encoder(
            EncoderGlobalRepInput(data=log_pose_trans_norm_factors_across_views)
        ).features
        # Zero out the pose trans scale features where the metric scale mask is False
        # Scale encoding is only provided for metric scale samples
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * metric_scale_pose_trans_mask.unsqueeze(-1)
        )

        # Fuse the pose quat features, pose trans features, pose trans scale features and pose trans type PE features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + pose_quats_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views
    
    
    def encode_rays_and_pose(
        self,
        num_views,
        batch_size_per_view,
        ray_dirs,
        camera_pose_quats,
        camera_pose_trans,
        is_metric_scale
    ):
        """
        Encode all the input optional geometric modalities and fuses it with the image encoder features in a single forward pass.
        Assumes all the input views have the same shape and batch size.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            all_encoder_features_across_views (List[torch.Tensor]): List of tensors containing the encoded image features for all N views.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        # Reshape inputs flattening batch_size_per_view and num_views
        ray_dirs = ray_dirs.reshape(batch_size_per_view*num_views, *ray_dirs.shape[2:]).contiguous()
        camera_pose_quats = camera_pose_quats.reshape(batch_size_per_view*num_views, *camera_pose_quats.shape[2:]).contiguous()
        camera_pose_trans = camera_pose_trans.reshape(batch_size_per_view*num_views, *camera_pose_trans.shape[2:]).contiguous()
        is_metric_scale = is_metric_scale.reshape(batch_size_per_view*num_views, *is_metric_scale.shape[2:]).contiguous()
        
        # Encode the ray directions
        ray_dirs_features_across_views = self.ray_dirs_encoder_trained(
            ViTEncoderNonImageInput(data=ray_dirs)
        ).features
        
        # Encode the pose quats
        pose_quats_features_across_views = self.cam_rot_encoder_trained(
            EncoderGlobalRepInput(data=camera_pose_quats)
        ).features

        # Get the metric scale mask for all samples
        device = ray_dirs.device
        metric_scale_pose_trans_mask = torch.zeros(
            (batch_size_per_view * num_views), dtype=torch.bool, device=device
        )
        for view_idx in range(num_views):
            metric_scale_mask = is_metric_scale[view_idx]
            metric_scale_pose_trans_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ] = metric_scale_mask

        # Get the scale norm factor for all the samples and scale the pose translations
        pose_trans_across_views = torch.split(
            camera_pose_trans, batch_size_per_view, dim=0
        )  # Split into num_views chunks
        pose_trans_across_views = torch.stack(
            pose_trans_across_views, dim=1
        )  # Stack the views along a new dimension (batch_size_per_view, num_views, 3)
        scaled_pose_trans_across_views, pose_trans_norm_factors = (
            normalize_pose_translations(
                pose_trans_across_views, return_norm_factor=True
            )
        )

        # Resize the pose translation back to (batch_size_per_view * num_views, 3) and extend the norm factor to (batch_size_per_view * num_views, 1)
        scaled_pose_trans_across_views = scaled_pose_trans_across_views.unbind(
            dim=1
        )  # Convert back to list of views, where each view has batch_size_per_view tensor
        scaled_pose_trans_across_views = torch.cat(
            scaled_pose_trans_across_views, dim=0
        )  # Concatenate back to (batch_size_per_view * num_views, 3)
        pose_trans_norm_factors_across_views = pose_trans_norm_factors.unsqueeze(
            -1
        ).repeat(num_views, 1)  # (B, ) -> (B * V, 1)

        # Encode the pose trans
        pose_trans_features_across_views = self.cam_trans_encoder_trained(
            EncoderGlobalRepInput(data=scaled_pose_trans_across_views)
        ).features

        # Encode the pose translation norm factors using the log scale encoder for pose trans
        log_pose_trans_norm_factors_across_views = torch.log(
            pose_trans_norm_factors_across_views + 1e-8
        )
        pose_trans_scale_features_across_views = self.cam_trans_scale_encoder_trained(
            EncoderGlobalRepInput(data=log_pose_trans_norm_factors_across_views)
        ).features
        # Zero out the pose trans scale features where the metric scale mask is False
        # Scale encoding is only provided for metric scale samples
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * metric_scale_pose_trans_mask.unsqueeze(-1)
        )

        # Fuse the pose quat features, pose trans features, pose trans scale features and pose trans type PE features with the other encoder features
        all_encoder_features_across_views = (
            ray_dirs_features_across_views
            + pose_quats_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        # Normalize the fused features (permute -> normalize -> permute)
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 2, 3, 1
        ).contiguous()
        all_encoder_features_across_views = self.fusion_norm_layer(
            all_encoder_features_across_views
        )
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 3, 1, 2
        ).contiguous()

        # Split the batched views into individual views
        fused_all_encoder_features_across_views = (
            all_encoder_features_across_views.chunk(num_views, dim=0)
        )

        return fused_all_encoder_features_across_views
    
    
    def _encode_and_fuse_geometric_inputs(
        self,
        num_views,
        batch_size_per_view,
        ray_dirs,
        camera_pose_quats,
        camera_pose_trans,
        all_encoder_features_across_views_list,
        is_metric_scale
    ):
        """
        Encode all the input optional geometric modalities and fuses it with the image encoder features in a single forward pass.
        Assumes all the input views have the same shape and batch size.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            all_encoder_features_across_views (List[torch.Tensor]): List of tensors containing the encoded image features for all N views.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        # Reshape inputs flattening batch_size_per_view and num_views
        ray_dirs = ray_dirs.reshape(batch_size_per_view*num_views, *ray_dirs.shape[2:]).contiguous()
        camera_pose_quats = camera_pose_quats.reshape(batch_size_per_view*num_views, *camera_pose_quats.shape[2:]).contiguous()
        camera_pose_trans = camera_pose_trans.reshape(batch_size_per_view*num_views, *camera_pose_trans.shape[2:]).contiguous()
        is_metric_scale = is_metric_scale.reshape(batch_size_per_view*num_views, *is_metric_scale.shape[2:]).contiguous()
        
        all_encoder_features_across_views = torch.cat(
            all_encoder_features_across_views_list, dim=0
        )

        # Encode the ray directions and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_ray_dirs(
            ray_dirs,
            all_encoder_features_across_views,
        )

        # Encode the cam quat and trans and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_cam_quats_and_trans(
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            camera_pose_quats,
            camera_pose_trans,
            is_metric_scale
        )

        # Normalize the fused features (permute -> normalize -> permute)
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 2, 3, 1
        ).contiguous()
        all_encoder_features_across_views = self.fusion_norm_layer(
            all_encoder_features_across_views
        )
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 3, 1, 2
        ).contiguous()

        # Split the batched views into individual views
        fused_all_encoder_features_across_views = (
            all_encoder_features_across_views.chunk(num_views, dim=0)
        )

        return fused_all_encoder_features_across_views
    
    
    def forward_feats(self, images, ray_dirs, camera_pose_quats, camera_pose_trans, is_metric_scale, memory_efficient_inference=False):
        """
        Forward pass performing the following operations:
        1. Encodes the N input views (images).
        2. Encodes the optional geometric inputs (ray directions, depths, camera rotations, camera translations).
        3. Fuses the encoded features from the N input views and the optional geometric inputs using addition and normalization.
        4. Information sharing across the encoded features and a scale token using a multi-view attention transformer.
        5. Passes the final features from transformer through the prediction heads.
        6. Returns the processed final outputs for N views.

        Assumption:
        - All the input views and dense geometric inputs have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W). Input images must be normalized based on the data norm type of image encoder.
                                    "data_norm_type" (list): [model.encoder.data_norm_type]
                                Optionally, each dictionary can also contain the following keys for the respective optional geometric inputs:
                                    "ray_directions_cam" (tensor): Ray directions in the local camera frame. Tensor of shape (B, H, W, 3).
                                    "depth_along_ray" (tensor): Depth along the ray. Tensor of shape (B, H, W, 1).
                                    "camera_pose_quats" (tensor): Camera pose quaternions. Tensor of shape (B, 4). Camera pose is opencv (RDF) cam2world transformation.
                                    "camera_pose_trans" (tensor): Camera pose translations. Tensor of shape (B, 3). Camera pose is opencv (RDF) cam2world transformation.
                                    "is_metric_scale" (tensor): Boolean tensor indicating whether the geometric inputs are in metric scale or not. Tensor of shape (B, 1).
            memory_efficient_inference (bool): Whether to use memory efficient inference or not. This runs the dense prediction head (the memory bottleneck) in a memory efficient manner. Default is False.

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        batch_size_per_view, num_views, c, h, w = images.shape
        
        # Run the image encoder on all the input views
        all_encoder_features_across_views = self._encode_n_views(num_views, images)
        
        # Encode the optional geometric inputs and fuse with the encoded features from the N input views
        # Use high precision to prevent NaN values after layer norm in dense representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_geometric_inputs(
                    num_views,
                    batch_size_per_view,
                    ray_dirs,
                    camera_pose_quats,
                    camera_pose_trans,
                    all_encoder_features_across_views,
                    is_metric_scale
                )
            )

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )
        if self.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
        elif self.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self.info_sharing(info_sharing_input)
        
        dense_head_inputs_list = []
        if self.use_encoder_features_for_dpt:
            # Stack all the image encoder features for all views
            stacked_encoder_features = torch.cat(
                all_encoder_features_across_views, dim=0
            )
            dense_head_inputs_list.append(stacked_encoder_features)
            # Stack the first intermediate features for all views
            stacked_intermediate_features_1 = torch.cat(
                intermediate_info_sharing_multi_view_feat[0].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_1)
            # Stack the second intermediate features for all views
            stacked_intermediate_features_2 = torch.cat(
                intermediate_info_sharing_multi_view_feat[1].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_2)
            # Stack the last layer features for all views
            stacked_final_features = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
            dense_head_inputs_list.append(stacked_final_features)
        else:
            # Stack the first intermediate features for all views
            stacked_intermediate_features_1 = torch.cat(
                intermediate_info_sharing_multi_view_feat[0].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_1)
            # Stack the second intermediate features for all views
            stacked_intermediate_features_2 = torch.cat(
                intermediate_info_sharing_multi_view_feat[1].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_2)
            # Stack the third intermediate features for all views
            stacked_intermediate_features_3 = torch.cat(
                intermediate_info_sharing_multi_view_feat[2].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_3)
            # Stack the last layer
            stacked_final_features = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
            dense_head_inputs_list.append(stacked_final_features)
        
        return dense_head_inputs_list
    
    
    @torch.inference_mode()
    def infer_feats(
        self,
        views: List[Dict[str, Any]],
        memory_efficient_inference: bool = False,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        apply_mask: bool = True,
        mask_edges: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10,
        ignore_calibration_inputs: bool = False,
        ignore_depth_inputs: bool = False,
        ignore_pose_inputs: bool = False,
        ignore_depth_scale_inputs: bool = False,
        ignore_pose_scale_inputs: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        User-friendly inference with strict input validation and automatic conversion.

        Args:
            views: List of view dictionaries. Each dict can contain:
                Required:
                - 'img': torch.Tensor of shape (B, 3, H, W) - normalized RGB images
                - 'data_norm_type': str - normalization type used to normalize the images (must be equal to self.model.encoder.data_norm_type)

                Optional Geometric Inputs (only one of intrinsics OR ray_directions):
                - 'intrinsics': torch.Tensor of shape (B, 3, 3) - will be converted to ray directions
                - 'ray_directions': torch.Tensor of shape (B, H, W, 3) - ray directions in camera frame
                - 'depth_z': torch.Tensor of shape (B, H, W, 1) - Z depth in camera frame (intrinsics or ray_directions must be provided)
                - 'camera_poses': torch.Tensor of shape (B, 4, 4) or tuple of (quats - (B, 4), trans - (B, 3)) - can be any world frame
                - 'is_metric_scale': bool or torch.Tensor of shape (B,) - if not provided, defaults to True

                Optional Additional Info:
                - 'instance': List[str] where length of list is B - instance info for each view
                - 'idx': List[int] where length of list is B - index info for each view
                - 'true_shape': List[tuple] where length of list is B - true shape info (H, W) for each view

            memory_efficient_inference: Whether to use memory-efficient inference for dense prediction heads (trades off speed). Defaults to False.
            use_amp: Whether to use automatic mixed precision for faster inference. Defaults to True.
            amp_dtype: The dtype to use for mixed precision. Defaults to "bf16" (bfloat16). Options: "fp16", "bf16", "fp32".
            apply_mask: Whether to apply the non-ambiguous mask to the output. Defaults to True.
            mask_edges: Whether to compute an edge mask based on normals and depth and apply it to the output. Defaults to True.
            edge_normal_threshold: Tolerance threshold for normals-based edge detection. Defaults to 5.0.
            edge_depth_threshold: Relative tolerance threshold for depth-based edge detection. Defaults to 0.03.
            apply_confidence_mask: Whether to apply the confidence mask to the output. Defaults to False.
            confidence_percentile: The percentile to use for the confidence threshold. Defaults to 10.
            ignore_calibration_inputs: Whether to ignore the calibration inputs (intrinsics and ray_directions). Defaults to False.
            ignore_depth_inputs: Whether to ignore the depth inputs. Defaults to False.
            ignore_pose_inputs: Whether to ignore the pose inputs. Defaults to False.
            ignore_depth_scale_inputs: Whether to ignore the depth scale inputs. Defaults to False.
            ignore_pose_scale_inputs: Whether to ignore the pose scale inputs. Defaults to False.

        IMPORTANT CONSTRAINTS:
        - Cannot provide both 'intrinsics' and 'ray_directions' (they represent the same information)
        - If 'depth' is provided, then 'intrinsics' or 'ray_directions' must also be provided
        - If ANY view has 'camera_poses', then view 0 (first view) MUST also have 'camera_poses'

        Returns:
            List of prediction dictionaries, one per view. Each dict contains:
                - 'img_no_norm': torch.Tensor of shape (B, H, W, 3) - denormalized rgb images
                - 'pts3d': torch.Tensor of shape (B, H, W, 3) - predicted points in world frame
                - 'pts3d_cam': torch.Tensor of shape (B, H, W, 3) - predicted points in camera frame
                - 'ray_directions': torch.Tensor of shape (B, H, W, 3) - ray directions in camera frame
                - 'intrinsics': torch.Tensor of shape (B, 3, 3) - pinhole camera intrinsics recovered from ray directions
                - 'depth_along_ray': torch.Tensor of shape (B, H, W, 1) - depth along ray in camera frame
                - 'depth_z': torch.Tensor of shape (B, H, W, 1) - Z depth in camera frame
                - 'cam_trans': torch.Tensor of shape (B, 3) - camera translation in world frame
                - 'cam_quats': torch.Tensor of shape (B, 4) - camera quaternion in world frame
                - 'camera_poses': torch.Tensor of shape (B, 4, 4) - camera pose in world frame
                - 'metric_scaling_factor': torch.Tensor of shape (B,) - applied metric scaling factor
                - 'mask': torch.Tensor of shape (B, H, W, 1) - combo of non-ambiguous mask, edge mask and confidence-based mask if used
                - 'non_ambiguous_mask': torch.Tensor of shape (B, H, W) - non-ambiguous mask
                - 'non_ambiguous_mask_logits': torch.Tensor of shape (B, H, W) - non-ambiguous mask logits
                - 'conf': torch.Tensor of shape (B, H, W) - confidence

        Raises:
            ValueError: For invalid inputs, missing required keys, conflicting modalities, or constraint violations
        """
        # Determine the mixed precision floating point type
        if use_amp:
            if amp_dtype == "fp16":
                amp_dtype = torch.float16
            elif amp_dtype == "bf16":
                if torch.cuda.is_bf16_supported():
                    amp_dtype = torch.bfloat16
                else:
                    warnings.warn(
                        "bf16 is not supported on this device. Using fp16 instead."
                    )
                    amp_dtype = torch.float16
            elif amp_dtype == "fp32":
                amp_dtype = torch.float32
        else:
            amp_dtype = torch.float32

        # Run the encoder
        with torch.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):
            dense_head_inputs_list = self.forward_feats(
                views, memory_efficient_inference=memory_efficient_inference
            )
            
        return dense_head_inputs_list
