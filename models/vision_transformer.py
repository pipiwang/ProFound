"""Vision Transformer model implementation based on timm.

https://github.com/huggingface/pytorch-image-models/blob/v0.9.16/timm/models/vision_transformer.py
https://github.com/TonyLianLong/CrossMAE/blob/main/transformer_utils.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from timm.layers import DropPath, Mlp, use_fused_attn
from timm.models.vision_transformer import LayerScale
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import TypedDict

from models.util import get_tokens, init_weights, checkpoint_seq

if TYPE_CHECKING:
    from torch.jit import Final


class Attention(nn.Module):
    """Attention layer supporting attention mask and different query and key."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """Initialize the attention."""
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            q: query tokens, (batch, n_q_tokens, ch).
            k: optional key tokens, (batch, n_k_tokens, ch), if None, use q for both query and key.

        Returns:
            q: query tokens, (batch, n_q_tokens, ch).
        """
        if k is None:
            k = q
        batch, n_q_tokens, ch = q.shape
        n_k_tokens = k.shape[1]
        q = (
            self.q(q)
            .reshape(batch, n_q_tokens, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(k)
            .reshape(batch, n_k_tokens, 2, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # q, k, v: (batch, n_heads, n_tokens, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            q = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            q = attn @ v

        q = q.transpose(1, 2).reshape(batch, n_q_tokens, ch)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q


class Block(nn.Module):
    """Vision Transformer block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        """Initialize the block."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, q: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            q: query tokens, (batch, n_q_tokens, ch), with positional and view embeddings added without norm.
            k: optional key tokens after norm, (batch, n_k_tokens, ch), if None, use q for both query and key.

        Returns:
            q: query tokens, (batch, n_q_tokens, ch).
        """
        q = q + self.drop_path1(self.ls1(self.attn(self.norm1(q), k)))
        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


class ViTEncoder(nn.Module):
    """VisionTransformer encoder."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: int,
        norm_layer: nn.Module,
        drop_path: float,
    ) -> None:
        """Initialize the module.

        Args:
            embed_dim: number of embedding channels.
            depth: number of layers.
            n_heads: number of heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_layer: normalization layer.
            drop_path: drop path rate.
        """
        super().__init__()
        self.grad_ckpt = False
        self.cls_token = get_tokens(embed_dim=embed_dim, n_tokens=1)
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ],
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tokens with positional embedding added, (batch, n_enc_keep, emb_dim).

        Returns:
            x: latent tensor, (batch, 1+n_enc_keep, emb_dim).
        """
        # append cls token
        # (batch, 1, emb_dim)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1).contiguous()
        # (batch, 1+n_enc_keep, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x = checkpoint_seq(self.blocks, x) if self.grad_ckpt else self.blocks(x)
        x = self.norm(x)
        return x

    def feature_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feature extraction, returns all intermediate features.

        Args:
            x: tokens with positional embedding added, (batch, n_enc_keep, emb_dim).

        Returns:
            x: stacked tensor, (batch, 1+n_enc_keep, emb_dim, n_layers).
        """
        cls_tokens = self.cls_token.expand(
            x.shape[0], -1, -1
        ).contiguous()  # (batch, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 1+n_enc_keep, emb_dim)
        xs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i != len(self.blocks) - 1:  # the last layer is not appended
                xs.append(x)
        x = self.norm(x)
        xs.append(x)
        return torch.stack(xs, dim=-1)  # (batch, 1+n_enc_keep, emb_dim, n_layers)


class ViTDecoder(nn.Module):
    """VisionTransformer decoder."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: int,
        norm_layer: nn.Module,
        drop_path: float,
    ) -> None:
        """Initialize the module.

        Args:
            embed_dim: number of embedding channels.
            depth: number of layers.
            n_heads: number of heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_layer: normalization layer.
            drop_path: drop path rate.
        """
        super().__init__()
        self.grad_ckpt = False
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ],
        )
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
        x_q: torch.Tensor,
        x_k: torch.Tensor | None,
        n_enc_masked: int,
    ) -> torch.Tensor:
        """Forward pass of the decoder.

        https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_pretrain.py
        https://github.com/TonyLianLong/CrossMAE/blob/main/models_cross.py for cross attention.

        Args:
            x_q: query patches.
              if not cross attention, it's all patched with masked ones at the end, (batch, 1+n_patches, dec_emb_dim).
              else, it's cls token and masked patches, (batch, 1+n_enc_masked, dec_emb_dim).
            x_k: key patches.
              if not cross attention, it's None.
              else, it's visible patches, (batch, 1+n_enc_keep, dec_emb_dim).
            n_enc_masked: number of masked patches.

        Returns:
            pred: predicted masked patches, (batch, n_enc_masked, dec_emb_dim).
        """
        # (batch, 1+n_patches, dec_emb_dim)
        for blk in self.blocks:
            x_q = (
                checkpoint(blk, x_q, x_k, use_reentrant=False)
                if self.grad_ckpt
                else blk(x_q, x_k)
            )

        # remove cls token, visible patches, return only masked patches
        # (batch, n_enc_masked, dec_emb_dim)
        x_q = x_q[:, -n_enc_masked:, :]
        x_q = self.norm(x_q)

        return x_q


class ViTConfig(TypedDict):
    """Configuration for MAE ViT encoder decoder."""

    enc_embed_dim: int
    enc_depth: int
    enc_n_heads: int
    dec_embed_dim: int
    dec_depth: int
    dec_n_heads: int


def get_vit_config(size: str) -> ViTConfig:
    """Get VisionTransformer configuration.

    Except from tiny, other configurations are from MAE.
    https://github.com/facebookresearch/mae/blob/main/models_mae.py

    Args:
        size: size of the model, must be in ['tiny', 'base', 'large', 'huge'].

    Returns:
        config_dict: configuration dictionary.
    """
    if size not in ["tiny", "base", "large", "huge"]:
        raise ValueError(
            f"size must be in ['tiny', 'base', 'large', 'huge'], got {size}."
        )
    return {
        "tiny": ViTConfig(
            enc_embed_dim=32,
            enc_depth=2,
            enc_n_heads=4,
            dec_embed_dim=32,
            dec_depth=1,
            dec_n_heads=4,
        ),
        "base": ViTConfig(
            enc_embed_dim=768,
            enc_depth=12,
            enc_n_heads=12,
            dec_embed_dim=384,
            dec_depth=8,
            dec_n_heads=16,
        ),
        "large": ViTConfig(
            enc_embed_dim=1024,
            enc_depth=24,
            enc_n_heads=16,
            dec_embed_dim=512,
            dec_depth=8,
            dec_n_heads=16,
        ),
        "huge": ViTConfig(
            enc_embed_dim=1280,
            enc_depth=32,
            enc_n_heads=16,
            dec_embed_dim=512,
            dec_depth=8,
            dec_n_heads=16,
        ),
    }[size]
