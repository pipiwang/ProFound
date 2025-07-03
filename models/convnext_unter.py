import torch.nn.functional as F
from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from models.util import LayerNorm


class ConvnextUNETR_Decoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 16,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
        hidden_size = [96, 192, 384, 768]
    ) -> None:

        super().__init__()

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size[0],
            out_channels=feature_size * 2,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size[1],
            out_channels=feature_size * 4,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size[2],
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size[3],
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward(self, x, x1, x2, x3, x4):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(x1)
        enc3 = self.encoder3(x2)
        enc4 = self.encoder4(x3)
        dec3 = self.decoder5(x4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        mask = self.out(out)
        return mask


class ConvnextUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convnext,
        feature_size: int = 16,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
        hidden_size = [96, 192, 384, 768]
    ) -> None:
        
        super().__init__()

        self.encoder = convnext

        self.norm1 = LayerNorm(hidden_size[0], eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(hidden_size[1], eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(hidden_size[2], eps=1e-6, data_format="channels_first")

        self.decoder = ConvnextUNETR_Decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            spatial_dims=spatial_dims,
            hidden_size=hidden_size
        )

    def forward(self, x):
        _, hidden_states_out = self.encoder(x, ret_hids=True)
        x1, x2, x3, x4 = hidden_states_out
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x3 = self.norm3(x3)
        x4 = x4.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x4 = self.encoder.norm(x4)
        x4 = x4.permute(0, 4, 1, 2, 3)
        mask = self.decoder(x, x1, x2, x3, x4)
        return mask
