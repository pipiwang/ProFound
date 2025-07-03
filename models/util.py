import torch
import torch.nn as nn
from itertools import chain
from typing import Callable
from torch.utils.checkpoint import checkpoint

import numpy.random as random
import torch.nn.functional as F
# from MinkowskiEngine import SparseTensor


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# class MinkowskiGRN(nn.Module):
#     """GRN layer for sparse tensors."""

#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, dim))

#     def forward(self, x):
#         cm = x.coordinate_manager
#         in_key = x.coordinate_map_key

#         Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return SparseTensor(
#             self.gamma * (x.F * Nx) + self.beta + x.F,
#             coordinate_map_key=in_key,
#             coordinate_manager=cm,
#         )


class MinkowskiDropPath(nn.Module):
    """Drop Path for sparse tensors."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = (
            torch.cat(
                [
                    (
                        torch.ones(len(_))
                        if random.uniform(0, 1) > self.drop_prob
                        else torch.zeros(len(_))
                    )
                    for _ in x.decomposed_coordinates
                ]
            )
            .view(-1, 1)
            .to(x.device)
        )
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
            x.F * mask, coordinate_map_key=in_key, coordinate_manager=cm
        )


class MinkowskiLayerNorm(nn.Module):
    """Channel-wise layer normalization for sparse tensors."""

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            if len(x.shape) == 3: # for vit adapter
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight * x + self.bias
                return x
            else:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
                return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def get_tokens(embed_dim: int, n_tokens: int) -> nn.Parameter:
    """Return a learnable token of shape (1, n_tokens, embed_dim).

    Args:
        embed_dim: number of embedding channels.
        n_tokens: number of tokens.

    Returns:
        token: learnable token.
    """
    token = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
    # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    nn.init.trunc_normal_(token, std=0.02, b=2.0)
    return token


def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


"""Gradient checkpointing utilities.

Copied from
https://github.com/huggingface/pytorch-image-models/blob/f8979d4f50b7920c78511746f7315df8f1857bc5/timm/models/_manipulate.py
and added use_reentrant=False following warnings in pytorch docs.
"""


def checkpoint_seq(
    functions: nn.Sequential,
    x: torch.Tensor,
    every: int = 1,
    flatten: bool = False,
    skip_last: bool = False,
    preserve_rng_state: bool = True,
) -> torch.Tensor:
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """

    def run_function(
        start: int, end: int, functions: nn.Sequential
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def forward(_x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(
            run_function(start, end, functions),
            x,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
        )
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x
