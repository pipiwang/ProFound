# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import pdb

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiChannelwiseConvolution,
    MinkowskiLinear,
)

from timm.models.layers import trunc_normal_
from models.convnextv2_sparse import SparseConvNeXtV2
from models.convnextv2 import Block


class FCMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone"""

    def __init__(
        self,
        img_size=(64, 256, 256),
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        decoder_depth=1,
        decoder_embed_dim=512,
        patch_size=(16, 16, 16),
        norm_pix_loss=False,
    ):
        super().__init__()

        # configs
        self.img_size = img_size
        self.depths = depths
        self.imds = dims
        self.patch_size = patch_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss

        # encoder
        self.encoder = SparseConvNeXtV2(
            in_chans=in_chans, depths=depths, dims=dims, D=3
        )
        # decoder
        self.proj = nn.Conv3d(
            in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1
        )
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1, 1))
        decoder = [
            Block(dim=decoder_embed_dim, drop_path=0.0) for i in range(decoder_depth)
        ]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv3d(
            in_channels=decoder_embed_dim,
            out_channels=self.patch_size[0]
            * self.patch_size[1]
            * self.patch_size[2]
            * in_chans,
            kernel_size=1,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiChannelwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv3d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        """
        imgs: (N, C, D, H, W)
        x: (N, L, patch_size**3 *3)
        """

        B, C, D, H, W = imgs.shape
        imgs = imgs.contiguous().reshape(
            B,
            C,
            self.grid_size[0],
            self.patch_size[0],
            self.grid_size[1],
            self.patch_size[1],
            self.grid_size[2],
            self.patch_size[2],
        )  # [B,C,gd,pd,gh,ph,gw,pw]
        imgs = (
            imgs.permute(0, 2, 4, 6, 3, 5, 7, 1)
            .contiguous()
            .reshape(
                B,
                self.grid_size[0] * self.grid_size[1] * self.grid_size[2],
                self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * C,
            )
        )  # [B,gh*gw*gd,ph*pw*pd*C]
        return imgs

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *3)
        imgs: (N, C, D, H, W)
        """
        B, L, dim = x.shape
        x = x.reshape(
            shape=(
                x.shape[0],
                self.grid_size[0],
                self.grid_size[1],
                self.grid_size[2],
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
                3,
            )
        )
        x = (
            x.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .contiguous()
            .reshape(
                B,
                3,
                self.grid_size[0] * self.patch_size[0],
                self.grid_size[1] * self.patch_size[1],
                self.grid_size[2] * self.patch_size[2],
            )
        )
        return x

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2  # N, L
        return (
            mask.reshape(-1, self.grid_size[0], self.grid_size[1], self.grid_size[2])
            .repeat_interleave(scale, axis=1)
            .repeat_interleave(scale, axis=2)
            .repeat_interleave(scale, axis=3)
        )

    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        num_stages = len(self.encoder.stages)
        x = self.encoder(imgs, self.upsample_mask(mask, 2 ** (num_stages - 2)))
        return x, mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, d, h, w = x.shape
        mask = mask.reshape(-1, d, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(
            x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]
        )
        x = x * (1.0 - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 5:
            n, c, _, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum("ncl->nlc", pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def convnextv2_atto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_pico(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = FCMAE(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = FCMAE(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


if __name__ == "main":
    model = convnextv2_base().cuda()
    x = torch.rand(1, 3, 256, 256, 32).cuda()
    print(model(x).shape)
