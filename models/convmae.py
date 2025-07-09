"""Convolutional Masked Autoencoder for images with VisionTransformer backbone.

https://github.com/Alpha-VL/ConvMAE/
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.conv import ConvNormActBlock, MaskedConvBlock
from models.patch import patchify
from models.patch_embed import PatchEmbed
from models.patch_masking import get_batch_random_patch_mask
from util.pos_embed import get_3d_sincos_pos_embed
from models.vision_transformer import ViTDecoder, ViTEncoder, get_vit_config
from models.util import init_weights, get_tokens


def upsample_mask(mask: torch.Tensor, scale_factor: tuple[int, ...]) -> torch.Tensor:
    """Upsample mask.

    Args:
        mask: binary mask, (batch, *spatial_shape).
        scale_factor: scale factor for each spatial dimension.

    Returns:
        mask: upsampled mask, (batch, *upsampled_spatial_shape).
    """
    if mask.ndim != len(scale_factor) + 1:
        raise ValueError(
            f"mask must have the same number of dimensions as scale_factor except batch, "
            f"got {mask.ndim} and {len(scale_factor)}."
        )
    expand_shape = (*(-1 for _ in range(mask.ndim)), math.prod(scale_factor))
    permute_dim: tuple[int, ...] = (0,)
    for i, _ in enumerate(scale_factor):
        permute_dim = (*permute_dim, i + 1, len(scale_factor) + i + 1)
    upsampled_shape = (
        mask.shape[0],
        *(s * f for s, f in zip(mask.shape[1:], scale_factor)),
    )
    mask = (
        mask.unsqueeze(-1)
        .expand(*expand_shape)
        .reshape(*mask.shape, *scale_factor)
        .permute(*permute_dim)
        .reshape(upsampled_shape)
    )
    return mask


class DownsampleEncoder(nn.Module):
    """Down-sample encoder module in ConvMAE before ViT."""

    def __init__(
        self,
        image_size: tuple[int, ...],
        in_chans: int,
        patch_size: tuple[int, ...],
        scale_factor: tuple[int, ...],
        conv_chans: list[int],
        conv_n_blocks: int,
        embed_dim: int,
        norm: str,
    ) -> None:
        """Initialize the module.

        Args:
            image_size: input image size.
            in_chans: number of input channels.
            patch_size: patch size for the first layer.
            scale_factor: scale factor for other layers.
            conv_chans: number of channels for each conv layer, if empty, no conv layers.
            conv_n_blocks: number of MaskedConvBlock for each conv_block.
            embed_dim: number of embedding channels for ViT encoder.
            norm: normalization layer, either 'instance' or 'layer' or 'group'.
        """
        super().__init__()
        self.grad_ckpt = False

        n_dims = len(image_size)
        conv_cls = nn.Conv2d if n_dims == 2 else nn.Conv3d
        n_conv_layers = len(conv_chans)
        self.patch_sizes = [patch_size] + [scale_factor] * n_conv_layers

        # shape pre-calculation
        grid_size: tuple[int, ...] = image_size
        for patch_size_i in self.patch_sizes:
            grid_size = tuple(s // p for s, p in zip(grid_size, patch_size_i))
        if min(grid_size) < 1:
            raise ValueError(
                f"Grid size {grid_size} is invalid, for {image_size} with {self.patch_sizes}."
            )

        # conv encoder
        conv_emb_size: tuple[int, ...] = image_size
        conv_emb_in_chans = in_chans
        self.conv_blocks = nn.ModuleList()
        for patch_size_i, chans_i in zip(self.patch_sizes[:-1], conv_chans):
            block = nn.Module()
            block.patch_embed = ConvNormActBlock(
                n_dims=n_dims,
                in_chans=conv_emb_in_chans,
                out_chans=chans_i,
                norm=norm,
                kernel_size=patch_size_i,
                stride=patch_size_i,
                padding="valid",
            )
            conv_emb_size = tuple(s // p for s, p in zip(conv_emb_size, patch_size_i))
            conv_emb_in_chans = chans_i

            block.conv = nn.ModuleList(
                [
                    MaskedConvBlock(n_dims=n_dims, in_chans=chans_i, norm=norm)
                    for _ in range(conv_n_blocks)
                ]
            )

            down_kernel_size = tuple(s // p for s, p in zip(conv_emb_size, grid_size))
            block.down = conv_cls(
                chans_i,
                embed_dim,
                kernel_size=down_kernel_size,
                stride=down_kernel_size,
                padding="valid",
            )
            self.conv_blocks.append(block)

        # embedding before ViT encoder
        self.patch_embed = PatchEmbed(
            img_size=conv_emb_size,
            patch_size=self.patch_sizes[-1],
            in_chans=conv_emb_in_chans,
            embed_dim=embed_dim,
        )
        self.linear = nn.Linear(
            embed_dim, embed_dim
        )  # original MAE does not have this layer

        self.pos_embed = nn.Parameter(
            torch.zeros(1, math.prod(self.patch_embed.grid_size), embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim, self.patch_embed.grid_size, cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing.

        Args:
            enable: whether to enable gradient checkpointing.
        """
        self.grad_ckpt = enable

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            image: (batch, in_chans, ...).
            mask: (batch, n_patches) at ViT grid size.

        Returns:
            skips: list of skipped features from each conv layer, each is (batch, chans, *spatial_shape).
            x: (batch, n_keep, emb_dim), input to ViT encoder.
        """
        batch_size = image.shape[0]

        # upsample mask to conv input resolution
        # each element is (batch, n_patches), 0 is keep, 1 is remove
        conv_masks: list[torch.Tensor | None] = []
        conv_mask = mask.reshape(batch_size, *self.patch_embed.grid_size)
        for patch_size in self.patch_sizes[:0:-1]:  # drop the first one and reverse
            conv_mask = upsample_mask(conv_mask, scale_factor=patch_size)
            conv_masks.insert(0, ~conv_mask)  # 1 is visible

        # conv encoder
        skips = []
        x = image
        for block, conv_mask in zip(self.conv_blocks, conv_masks):
            x = (
                checkpoint(block.patch_embed, x, use_reentrant=False)
                if self.grad_ckpt
                else block.patch_embed(x)
            )
            for conv in block.conv:
                x = (
                    checkpoint(conv, x, conv_mask, use_reentrant=False)
                    if self.grad_ckpt
                    else conv(x, conv_mask)
                )
            skips.append(x)

        # patch embedding
        # (batch, n_patches, emb_dim)
        x = self.linear(self.patch_embed(x)) + self.pos_embed

        # masking
        # (batch, n_keep, emb_dim)
        x = x[~mask].reshape(batch_size, -1, x.shape[-1])

        return skips, x

    def feature_forward(
        self,
        image: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            image: (batch, in_chans, ...).

        Returns:
            skips: list of skipped features from each conv layer, each is (batch, chans, *spatial_shape).
            x: (batch, n_patches, emb_dim), input to ViT encoder.
        """
        batch_size = image.shape[0]

        # conv encoder
        skips = []
        x = image
        conv_mask = None
        for block in self.conv_blocks:
            x = (
                checkpoint(block.patch_embed, x, use_reentrant=False)
                if self.grad_ckpt
                else block.patch_embed(x)
            )
            for conv in block.conv:
                x = (
                    checkpoint(conv, x, conv_mask, use_reentrant=False)
                    if self.grad_ckpt
                    else conv(x, conv_mask)
                )
            skips.append(x)

        # patch embedding
        # (batch, n_patches, emb_dim)
        x = self.linear(self.patch_embed(x)) + self.pos_embed

        # masking
        # (batch, n_patches, emb_dim)
        x = x.reshape(batch_size, -1, x.shape[-1])

        return skips, x


class MultiScaleFusion(nn.Module):
    """Multi-scale fusion module in ConvMAE."""

    def __init__(
        self,
        image_size: tuple[int, ...],
        patch_size: tuple[int, ...],
        scale_factor: tuple[int, ...],
        conv_chans: list[int],
        embed_dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """Initialize the module.

        Args:
            image_size: input image size.
            patch_size: patch size for the first layer.
            scale_factor: scale factor for other layers.
            conv_chans: number of channels for each conv layer, if empty, no conv layers.
            embed_dim: number of embedding channels for ViT encoder.
            norm_layer: normalization layer.
        """
        super().__init__()
        self.grad_ckpt = False

        n_dims = len(image_size)
        conv_cls = nn.Conv2d if n_dims == 2 else nn.Conv3d
        patch_sizes = [patch_size] + [scale_factor] * len(conv_chans)

        # shape pre-calculation
        grid_size: tuple[int, ...] = image_size
        for patch_size_i in patch_sizes:
            grid_size = tuple(s // p for s, p in zip(grid_size, patch_size_i))

        # downsample blocks
        conv_emb_size: tuple[int, ...] = image_size
        self.down_convs = nn.ModuleList()
        for i, ch in enumerate(conv_chans):
            conv_emb_size = tuple(s // p for s, p in zip(conv_emb_size, patch_sizes[i]))
            down_kernel_size = tuple(s // p for s, p in zip(conv_emb_size, grid_size))
            conv = conv_cls(
                ch,
                embed_dim,
                kernel_size=down_kernel_size,
                stride=down_kernel_size,
                padding="valid",
            )
            self.down_convs.append(conv)
        self.norm = norm_layer(embed_dim)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing.

        Args:
            enable: whether to enable gradient checkpointing.
        """
        self.grad_ckpt = enable

    def forward(
        self,
        skips: list[torch.Tensor],
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            skips: list of skipped features from each conv layer, each is (batch, chans, *spatial_shape).
            x: (batch, n_keep, emb_dim), output from ViT encoder.
            mask: (batch, n_patches) at ViT grid size.

        Returns:
            x: (batch, n_keep, emb_dim).
        """
        for skip, conv in zip(skips, self.down_convs):
            # (batch, emb_dim, *spatial_shape)
            down = (
                checkpoint(conv, skip, use_reentrant=False)
                if self.grad_ckpt
                else conv(skip)
            )
            down = down.flatten(2).transpose(1, 2)  # (batch, n_patches, emb_dim)
            # (batch, n_keep, emb_dim)
            down = down[~mask].reshape(x.shape[0], -1, x.shape[-1])
            x = x + down
        x = self.norm(x)
        return x


def get_decoder_patch_size(
    image_size: tuple[int, ...],
    n_conv_layers: int,
    enc_patch_size: tuple[int, ...],
    enc_scale_factor: tuple[int, ...],
) -> tuple[int, ...]:
    """Get decoder patch size based on encoder settings.

    Args:
        image_size: input image size.
        n_conv_layers: number of conv layers in encoder.
        enc_patch_size: patch size for the first layer.
        enc_scale_factor: scale factor for other layers.

    Returns:
        dec_patch_size: patch size for the top layer in decoder.
    """
    dec_patch_size = (1,) * len(image_size)
    for i in range(1 + n_conv_layers):
        patch_size = enc_patch_size if i == 0 else enc_scale_factor
        dec_patch_size = tuple(s * p for s, p in zip(dec_patch_size, patch_size))
    return dec_patch_size


def add_pos_embed_and_append_mask_token(
    x_vis: torch.Tensor,
    enc_mask: torch.Tensor,
    dec_pos_embed: nn.Parameter,
    mask_token: nn.Parameter,
    concat: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Add mask tokens and position embeddings.
    Args:
        x_vis: visible tokens without class token, (batch, n_enc_keep, dec_emb_dim).
        enc_mask: binary mask, (batch, n_patches),
            0 is keep/visible to encoder, 1 is remove.
        dec_pos_embed: positional embedding for decoder, (n_patches, dec_emb_dim).
        mask_token: learnable mask token, (1, 1, dec_emb_dim).
        concat: whether to concatenate mask tokens to the sequence.

    Returns:
        if concat:
            x: tensor with mask tokens and position embeddings, (batch, 1+n_patches, dec_emb_dim).
        else:
            x_vis: visible tokens with position embeddings, (batch, n_enc_keep, dec_emb_dim).
            x_mask: mask tokens with position embeddings, (batch, n_enc_masked, dec_emb_dim).
    """
    batch, n_enc_keep, dec_emb_dim = x_vis.shape
    _, n_patches = enc_mask.shape
    n_enc_masked = n_patches - n_enc_keep

    # shuffle the pos embedding
    dec_pe = dec_pos_embed.expand(
        batch, -1, -1
    ).contiguous()  # (batch, n_patches, dec_emb_dim)
    vis_pe = dec_pe[~enc_mask].reshape(
        batch, n_enc_keep, dec_emb_dim
    )  # (batch, n_enc_keep, dec_emb_dim)
    mask_pe = dec_pe[enc_mask].reshape(
        batch, n_enc_masked, dec_emb_dim
    )  # (batch, n_enc_masked, dec_emb_dim)

    # append mask tokens to sequence
    if concat:
        return torch.cat(
            [x_vis + vis_pe, mask_token + mask_pe], dim=1
        )  # (batch, n_patches, dec_emb_dim)
    return x_vis + vis_pe, mask_token + mask_pe


class DecoderEmbedding(nn.Module):
    """Decoder embedding module."""

    def __init__(
        self,
        enc_grid_size: tuple[int, ...],
        dec_embed_dim: int,
        add_embed_token: bool,
    ) -> None:
        """Initialize the module.

        Args:
            enc_grid_size: grid size of encoder.
            dec_embed_dim: number of embedding channels for decoder.
            add_embed_token: whether to add an embedding to all tokens.
        """
        super().__init__()

        self.pos_embed = nn.Parameter(
            torch.zeros(1, math.prod(enc_grid_size), dec_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            dec_embed_dim, enc_grid_size, cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.embed_token = (
            get_tokens(embed_dim=dec_embed_dim, n_tokens=1) if add_embed_token else None
        )
        self.mask_token = get_tokens(embed_dim=dec_embed_dim, n_tokens=1)

    def forward(
        self,
        x: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 1+n_enc_keep, dec_emb_dim).
            enc_mask: (batch, n_patches), 0 is keep/visible to encoder, 1 is remove.

        Returns:
            x_vis: visible tokens with position embeddings, (batch, n_enc_keep, dec_emb_dim).
            x_mask: mask tokens with position embeddings, (batch, n_enc_masked, dec_emb_dim).
        """
        x_vis, x_mask = add_pos_embed_and_append_mask_token(
            x_vis=x,
            enc_mask=enc_mask,
            dec_pos_embed=self.pos_embed,
            mask_token=self.mask_token,
            concat=False,
        )
        if self.embed_token is not None:
            x_vis = x_vis + self.embed_token
            x_mask = x_mask + self.embed_token
        return x_vis, x_mask


class ImageConvMaskedAutoencoder(nn.Module):
    """Masked autoencoder with convolutions for images."""

    def __init__(
        self,
        image_size: tuple[int, ...],
        in_chans: int,
        patch_size: tuple[int, ...],
        scale_factor: tuple[int, ...],
        conv_chans: list[int],
        conv_n_blocks: int,
        enc_embed_dim: int,
        enc_depth: int,
        enc_n_heads: int,
        dec_embed_dim: int,
        dec_depth: int,
        dec_n_heads: int,
        mlp_ratio: int = 4,
        norm_target: bool = False,
        cross_attn: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        drop_path: float = 0.0,
        norm: str = "layer",
    ) -> None:
        """Initialize the module.

        Args:
            image_size: input image size.
            in_chans: number of input channels.
            patch_size: patch size for the first layer.
            scale_factor: scale factor for other layers.
            conv_chans: number of channels for each conv layer, if empty, no conv layers.
            conv_n_blocks: number of MaskedConvBlock for each enc_conv_block.
            enc_embed_dim: number of embedding channels for ViT encoder.
            enc_depth: number of layers for ViT encoder.
            enc_n_heads: number of heads for ViT encoder.
            dec_embed_dim: number of embedding channels for ViT decoder.
            dec_depth: number of layers for ViT decoder.
            dec_n_heads: number of heads for ViT decoder.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_layer: normalization layer.
            norm_target: whether to normalize target values for loss.
            cross_attn: whether to use cross attention.
            drop_path: drop path rate, only used for fine-tuning.
            norm: normalization layer, either 'instance' or 'layer' or 'group'.
        """
        super().__init__()

        self.grad_ckpt = False

        self.norm_target = norm_target

        # encoder
        self.enc_down = DownsampleEncoder(
            image_size=image_size,
            in_chans=in_chans,
            patch_size=patch_size,
            scale_factor=scale_factor,
            conv_chans=conv_chans,
            conv_n_blocks=conv_n_blocks,
            embed_dim=enc_embed_dim,
            norm=norm,
        )
        self.enc_fusion = MultiScaleFusion(
            image_size=image_size,
            patch_size=patch_size,
            scale_factor=scale_factor,
            conv_chans=conv_chans,
            embed_dim=enc_embed_dim,
            norm_layer=norm_layer,
        )
        self.encoder = ViTEncoder(
            embed_dim=enc_embed_dim,
            depth=enc_depth,
            n_heads=enc_n_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            drop_path=drop_path,
        )

        # decoder embedding
        self.dec_linear = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.dec_embed = DecoderEmbedding(
            enc_grid_size=self.enc_down.patch_embed.grid_size,
            dec_embed_dim=dec_embed_dim,
            add_embed_token=False,
        )

        # decoder
        self.cross_attn = cross_attn
        self.decoder = ViTDecoder(
            embed_dim=dec_embed_dim,
            depth=dec_depth,
            n_heads=dec_n_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            drop_path=drop_path,
        )

        # prediction head
        self.dec_patch_size = get_decoder_patch_size(
            image_size=image_size,
            n_conv_layers=len(conv_chans),
            enc_patch_size=patch_size,
            enc_scale_factor=scale_factor,
        )
        self.pred_head = nn.Linear(
            dec_embed_dim, math.prod(self.dec_patch_size) * in_chans
        )

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing.

        Args:
            enable: whether to enable gradient checkpointing.
        """
        self.grad_ckpt = enable
        self.enc_down.set_grad_ckpt(enable)
        self.enc_fusion.set_grad_ckpt(enable)
        self.encoder.set_grad_ckpt(enable)
        self.decoder.set_grad_ckpt(enable)

    def forward(
        self,
        image: torch.Tensor,
        enc_mask_ratio: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            image: (batch, in_chans, ...).
            enc_mask_ratio: masking ratio for encoder.

        Returns:
            loss: MSE loss on masked patches.
            pred: predicted patches, (batch, n_patches, out_chains).
            enc_mask: binary mask, (batch, n_patches), 0 is keep/visible to encoder, 1 is remove.
            metrics: metrics, each value is a scalar tensor.
        """
        batch_size = image.shape[0]
        device = image.device

        # sample mask at ViT input resolution
        # (batch, n_patches), 0 is keep, 1 is remove
        enc_mask = get_batch_random_patch_mask(
            batch_size=batch_size,
            n_patches=self.enc_down.patch_embed.num_patches,
            mask_ratio=enc_mask_ratio,
            device=device,
        )
        # downsample image and masking
        # (batch, n_keep, enc_emb_dim)
        skips, x = self.enc_down(image, enc_mask)

        # encoder
        # (batch, 1+n_enc_keep, enc_emb_dim)
        x = self.encoder(x)

        # fuse skipped features
        # (batch, 1+n_enc_keep, enc_emb_dim)
        x = torch.cat(
            [x[:, :1, :], self.enc_fusion(skips, x[:, 1:, :], enc_mask)], dim=1
        )
        n_enc_masked = enc_mask.shape[1] - x.shape[1] + 1

        # project to decoder space, add position embedding and append mask token
        # (batch, 1+n_enc_keep, dec_emb_dim)
        x = self.dec_linear(x)

        # add position embedding and append mask token
        # (batch, n, dec_emb_dim)
        x_vis, x_mask = self.dec_embed(x[:, 1:, :], enc_mask=enc_mask)

        # decoder
        # (batch, n_enc_masked, dec_emb_dim)
        if self.cross_attn:
            x_q = torch.cat([x[:, :1, :], x_mask], dim=1)
            x = self.decoder(x_q, x_vis, n_enc_masked)
        else:
            x = torch.cat([x[:, :1, :], x_vis, x_mask], dim=1)
            x = self.decoder(x, None, n_enc_masked)

        # loss
        # (batch, n_enc_masked, out_chans)
        pred = self.pred_head(x)
        loss, metrics = mse_loss(
            target=patchify(image=image, patch_size=self.dec_patch_size),
            pred=pred,
            enc_mask=enc_mask,
            norm_target=self.norm_target,
        )
        return loss, pred, enc_mask


def mse_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    enc_mask: torch.Tensor,
    norm_target: bool,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward pass of the loss, only calculated on masked patches.

    TODO: https://arxiv.org/abs/2401.15900 add weight per patch

    Args:
        target: target patches, (batch, n_patches, out_chans).
        pred: predicted patches, (batch, n_enc_masked, out_chans).
        enc_mask: binary mask, (batch, n_patches),
            0 is keep/visible to encoder, 1 is to be predicted.
        norm_target: whether to normalize target values for loss.
        epsilon: small value to avoid division by zero.

    Returns:
        loss: MSE loss on masked patches.
        metrics: metrics.
    """
    metrics: dict[str, torch.Tensor] = {}
    mean = target.mean(dim=-1, keepdim=True)  # (batch, n_patches, 1)
    var = target.var(dim=-1, keepdim=True)
    std = var**0.5
    metrics.update(
        {
            "target_mean": mean.mean(),
            "target_std": std.mean(),
        }
    )
    if norm_target:
        target = (target - mean) / (std + epsilon)
    target = target[enc_mask].reshape(pred.shape)  # (batch, n_enc_masked, out_chans)

    loss = nn.MSELoss(reduction="none")(
        pred, target.detach()
    )  # squared error, (batch, n_enc_masked, out_chans)
    loss = loss.mean()  # scalar
    metrics["mse_loss"] = loss

    if norm_target and target.shape[1] > 0:
        # when normalizing target
        # pred_max is a good indicator of whether the model is learning
        metrics["normed_target_max"] = target.max()
        metrics["pred_max"] = pred.max()

    return loss, metrics


def convmae_vit_base():
    vit_config = get_vit_config("base")
    model = ImageConvMaskedAutoencoder(
        image_size=(64, 256, 256),
        in_chans=3,
        patch_size=(16, 8, 8),
        scale_factor=[1, 2, 2],
        conv_chans=[64, 128],
        conv_n_blocks=2,
        **vit_config,
    )
    return model
