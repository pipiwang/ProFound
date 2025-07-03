from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from models.util import LayerNorm, GRN

class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first") # nn.BatchNorm3d(out_channels)
        self.activation = nn.GELU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool3d(pool_scale),
            UperNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class UperNetPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    def __init__(
        self,
        pool_scales: Tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(
                pool_scale=pool_scale, in_channels=in_channels, channels=channels
            )
            self.blocks.append(block)
            self.add_module(str(i), block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="trilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UperNetHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """

    def __init__(self, in_channels, pool_scales, hidden_size, out_channels):
        super().__init__()
        self.pool_scales = pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = in_channels
        self.channels = hidden_size
        self.align_corners = False
        self.classifier = nn.Conv3d(self.channels, out_channels, kernel_size=1)

        # PSP Module
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = UperNetConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = UperNetConvModule(
                self.channels, self.channels, kernel_size=3, padding=1
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv3d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # build laterals
        laterals = [
            lateral_conv(encoder_hidden_states[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i],
                size=prev_shape,
                mode="trilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="trilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class UperNetFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        num_convs,
        out_channels,
        concat_input=False,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels[in_index]
        self.channels = hidden_size
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            UperNetConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        self.classifier = nn.Conv3d(self.channels, out_channels, kernel_size=1)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv3d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


class ViTAdapter(nn.Module):
    def __init__(
        self,
        img_size=(64, 256, 256),
        patch_size=(16, 32, 32),
        embed_dim=768,
        # out_indices=[3, 5, 7, 11],
    ):
        super().__init__()
        # self.out_indices = out_indices

        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = embed_dim

        if patch_size == (16, 32, 32):
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
                ),
                nn.BatchNorm3d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm3d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            # 8
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
                ),
                nn.BatchNorm3d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            # 16
            self.fpn3 = nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
                ),
            )

            # 32
            self.fpn4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

            self.adapters = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]

    def proj_feat(self, x):
        
        new_view = (x.size(0), *self.grid_size, self.hidden_size)
        # print(f"x.shape: {x.shape}, expected: {new_view}, grid_size: {self.grid_size}")
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(
            d + 1 for d in range(len(self.grid_size))
        )
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, encoder_hidden_states):
        output = []
        # print(f"len_encoder_hidden: {len(encoder_hidden_states)}")
        for index, op in zip(range(len(encoder_hidden_states)), self.adapters):
            output.append(op(self.proj_feat(encoder_hidden_states[index])))
        return output


class UperNet(nn.Module):
    def __init__(
        self,
        encoder,
        in_channels,
        out_channels,
        adapter=None,
        out_indices=None,
        pool_scales=[1, 2, 3, 6],
        hidden_size=512,
        auxiliary_channels=256,
        use_auxiliary_head=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.out_indices = out_indices
        self.decode_head = UperNetHead(
            in_channels=in_channels,
            pool_scales=pool_scales,
            hidden_size=hidden_size,
            out_channels=out_channels,
        )
        self.auxiliary_head = (
            UperNetFCNHead(
                in_channels=in_channels,
                hidden_size=auxiliary_channels,
                num_convs=1,
                out_channels=out_channels,
            )
            if use_auxiliary_head
            else None
        )

        self.hidden_norm = nn.ModuleList()
        for in_channel in in_channels:
            norm = LayerNorm(in_channel, eps=1e-6, data_format="channels_first") # nn.BatchNorm3d(out_channels)        
            self.hidden_norm.append(norm)

    def forward(self, x):
        # print(f"403 input x.shape: {x.shape}")
        encoder_hidden_states = self.encoder(x, ret_hids=True)
        # print(f"405 {type(encoder_hidden_states)}, encoder_hidden_states: {len(encoder_hidden_states)}")
        # for i, hidden_state in enumerate(encoder_hidden_states):
        #     print(f"407 encoder_hidden_states[{i}]: {type(hidden_state)}, {len(hidden_state)}")
        if isinstance(encoder_hidden_states, list) or isinstance(
            encoder_hidden_states, Tuple
        ):
            encoder_hidden_states = encoder_hidden_states[-1]
        # print(f"410 {type(encoder_hidden_states)}, encoder_hidden_states: {len(encoder_hidden_states)}")
        # for i, hidden_state in enumerate(encoder_hidden_states):
        #     print(f"412 encoder_hidden_states[{i}]: {hidden_state.shape}")
        if self.out_indices:
            encoder_hidden_states = [
                encoder_hidden_states[i] for i in self.out_indices
            ]
        
        encoder_hidden_states = [
            norm(encoder_hidden_states[i])
            for i, norm in enumerate(self.hidden_norm)
        ]
        # print(f"415 encoder_hidden_states: {len(encoder_hidden_states)}")
        # for i in range(len(encoder_hidden_states)):
        #     print(f"417 encoder_hidden_states[{i}]: {encoder_hidden_states[i].shape}")

        if self.adapter:
            encoder_hidden_states = self.adapter(encoder_hidden_states)

        logits = self.decode_head(encoder_hidden_states)
        logits = nn.functional.interpolate(
            logits, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        if not self.training:
            return logits

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(encoder_hidden_states)
            auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            return [logits, auxiliary_logits]
        return logits
