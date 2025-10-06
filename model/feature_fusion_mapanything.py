import torch
import torch.nn as nn
from typing import Union, Tuple, Iterable, List, Optional
from torch.utils.checkpoint import checkpoint

from uniception.models.libs.croco.dpt_block import make_fusion_block, make_nonlinearity, make_scratch, pair


class DPTFeatureFusion(nn.Module):
    """
    DPT head implementation without any transpose convs / explicit upsampling.

    Behavior change: this implementation does NOT perform any upsampling in the
    'input_process' stage. The ConvTranspose2d layers and stride-2 conv were
    replaced by Conv2d layers that preserve spatial resolution.
    """
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        main_tasks: Iterable[str] = ("rgb",),
        hooks: List[int] = [2, 5, 8, 11],
        input_feature_dims: Optional[Union[int, List[int]]] = 768,
        layer_dims: List[int] = [96, 192, 384, 768],
        feature_dim: int = 256,
        use_bn: bool = False,
        output_width_ratio=1,
        pretrained_checkpoint_path: str = None,
        checkpoint_gradient: bool = False,
        nonlinearity: str = "relu",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.checkpoint_gradient = checkpoint_gradient

        if isinstance(input_feature_dims, int):
            input_feature_dims = 4 * [input_feature_dims]
        else:
            input_feature_dims = input_feature_dims
            assert isinstance(input_feature_dims, List) and len(input_feature_dims) == 4

        self.input_feature_dims = input_feature_dims

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)

        # delete resconfunit1 in refinement 4 because it is not used, and will cause error in DDP.
        if hasattr(self.scratch.refinenet4, "resConfUnit1"):
            del self.scratch.refinenet4.resConfUnit1

        if self.input_feature_dims is not None:
            self.init(input_feature_dims=input_feature_dims)

        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DPT dense feature head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def init(self, input_feature_dims: Union[int, List[int]] = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.

        IMPORTANT: this variant removes transpose convolutions / upsampling.
        All postprocess modules now preserve spatial dimensions.
        """
        if isinstance(input_feature_dims, int):
            input_feature_dims = 4 * [input_feature_dims]

        # account for multi-task concatenation
        self.input_feature_dims = [dt * len(self.main_tasks) for dt in input_feature_dims]

        # 1) act_1_postprocess: previously had ConvTranspose2d(k=4,s=4) -> removed.
        #    Now: project channels only, preserve HxW.
        act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # keep a small conv to allow local mixing but avoid any up/down-sampling
            nn.Conv2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        # 2) act_2_postprocess: previously had ConvTranspose2d(k=2,s=2) -> removed.
        act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        # 3) act_3_postprocess: unchanged (already a 1x1 proj)
        act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[2],
                out_channels=self.layer_dims[2],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        # 4) act_4_postprocess: previously had a conv with stride=2 (downsample).
        #    Replace with a stride=1 conv so spatial dims are preserved.
        act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3,
                stride=1,   # changed from 2 -> 1 to avoid spatial downsampling
                padding=1,
            ),
        )

        act_postprocess = [act_1_postprocess, act_2_postprocess, act_3_postprocess, act_4_postprocess]

        # wrap each postprocess with the corresponding layer_rn from scratch (same behaviour as previous)
        self.input_process = nn.ModuleList(
            [nn.Sequential(act_, layer_rn_) for act_, layer_rn_ in zip(act_postprocess, self.scratch.layer_rn)]
        )

    def forward(self, layered_feats: List) -> torch.Tensor:
        """
        Forward pass (no upsampling performed here).
        """
        assert self.input_feature_dims is not None, "Need to call init(input_feature_dims) function first"

        # check input dimensions
        for hook_idx, hook in enumerate(self.hooks):
            assert (
                layered_feats[hook].shape[1] == self.input_feature_dims[hook_idx]
            ), f"Input feature dimension mismatch at hook {hook}. Expected BCHW"

        if not self.checkpoint_gradient:
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [layered_feats[hook] for hook in self.hooks]

            # Project layers to chosen feature dim (no spatial upsampling)
            layers = [self.input_process[idx](l) for idx, l in enumerate(layers)]

            # Fuse layers using refinement stages
            path_4 = self.scratch.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
            print('4', path_4.shape)
            path_3 = self.scratch.refinenet3(path_4, layers[2])
            print('3', path_3.shape)
            path_2 = self.scratch.refinenet2(path_3, layers[1])
            print('2', path_3.shape)
            feature_fused = self.scratch.refinenet1(path_2, layers[0])
            print('1', feature_fused.shape)
            raise
        else:
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [layered_feats[hook] for hook in self.hooks]

            layers = [checkpoint(self.input_process[idx], l, use_reentrant=False) for idx, l in enumerate(layers)]

            path_4 = checkpoint(self.scratch.refinenet4, layers[3], use_reentrant=False)[
                :, :, : layers[2].shape[2], : layers[2].shape[3]
            ]
            path_3 = checkpoint(self.scratch.refinenet3, path_4, layers[2], use_reentrant=False)
            path_2 = checkpoint(self.scratch.refinenet2, path_3, layers[1], use_reentrant=False)
            feature_fused = checkpoint(self.scratch.refinenet1, path_2, layers[0], use_reentrant=False)

        return feature_fused
