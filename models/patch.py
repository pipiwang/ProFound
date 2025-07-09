"""Patchify input images and unpatchify them back to the original shape.

Contiguous is applied after einsum, as suggested in
https://github.com/pytorch/pytorch/issues/47163#issuecomment-1766472400
"""

from __future__ import annotations

import math

import torch


def patchify_2d(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, H, W)
        patch_size: patch size (p, q).

    Returns:
        x: (batch, h*w, p*q*in_chans), where (p, q) is patch size,
            and H = h * p, W = w * q.
    """
    batch, in_chans, h, w = image.shape
    p, q = patch_size
    if h % p != 0:
        raise ValueError(f"Input height ({h}) cannot be divided by patch size ({p}).")
    if w % q != 0:
        raise ValueError(f"Input width ({w}) cannot be divided by patch size ({q}).")
    h, w = h // p, w // q  # grid size
    x = image.reshape(shape=(batch, in_chans, h, p, w, q))
    x = torch.einsum("nchpwq->nhwpqc", x).contiguous()
    x = x.reshape(shape=(batch, h * w, p * q * in_chans))
    return x


def patchify_3d(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, H, W, D)
        patch_size: patch size (p, q, r).

    Returns:
        x: (batch, h*w*d, p*q*r*in_chans), where (p, q, r) is patch size,
            and H = h * p, W = w * q, D = d * r.
    """
    batch, in_chans, h, w, d = image.shape
    p, q, r = patch_size
    if h % p != 0:
        raise ValueError(f"Input height ({h}) cannot be divided by patch size ({p}).")
    if w % q != 0:
        raise ValueError(f"Input width ({w}) cannot be divided by patch size ({q}).")
    if d % r != 0:
        raise ValueError(f"Input depth ({d}) cannot be divided by patch size ({r}).")
    h, w, d = h // p, w // q, d // r  # grid size
    x = image.reshape(shape=(batch, in_chans, h, p, w, q, d, r))
    x = torch.einsum("nchpwqdr->nhwdpqrc", x).contiguous()
    x = x.reshape(shape=(batch, h * w * d, p * q * r * in_chans))
    return x


def patchify_4d(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, H, W, D, T)
        patch_size: patch size (p, q, r, s).

    Returns:
        x: (batch, h*w*d*t, p*q*r*s*in_chans), where (p, q, r, s) is patch size,
            and H = h * p, W = w * q, D = d * r, T = t * s.
    """
    batch, in_chans, h, w, d, t = image.shape
    p, q, r, s = patch_size
    if h % p != 0:
        raise ValueError(f"Input height ({h}) cannot be divided by patch size ({p}).")
    if w % q != 0:
        raise ValueError(f"Input width ({w}) cannot be divided by patch size ({q}).")
    if d % r != 0:
        raise ValueError(f"Input depth ({d}) cannot be divided by patch size ({r}).")
    if t % s != 0:
        raise ValueError(f"Input time ({t}) cannot be divided by patch size ({s}).")
    h, w, d, t = h // p, w // q, d // r, t // s  # grid size
    x = image.reshape(shape=(batch, in_chans, h, p, w, q, d, r, t, s))
    x = torch.einsum("nchpwqdrts->nhwdtpqrsc", x).contiguous()
    x = x.reshape(shape=(batch, h * w * d * t, p * q * r * s * in_chans))
    return x


def patchify(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, ...).
        patch_size: corresponding patch size.

    Returns:
        x: (batch, n_patches, out_chans).
    """
    if len(patch_size) == 2:
        return patchify_2d(image, patch_size)
    if len(patch_size) == 3:
        return patchify_3d(image, patch_size)
    if len(patch_size) == 4:
        return patchify_4d(image, patch_size)
    raise ValueError(
        f"Patchify only supports 2D, 3D, and 4D images, got {len(patch_size)}D."
    )


def unpatchify_2d(
    x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]
) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, h*w, p*q*c), where (p, q) is patch size,
            and H = h * p, W = w * q.
        patch_size: patch size (p, q).
        grid_size: grid size (h, w).

    Returns:
        image of shape (batch, in_chans, H, W)
    """
    batch = x.shape[0]
    p, q = patch_size
    h, w = grid_size
    x = x.reshape(shape=(batch, h, w, p, q, -1))
    x = torch.einsum("nhwpqc->nchpwq", x).contiguous()
    x = x.reshape(shape=(batch, -1, h * p, w * q))
    return x


def unpatchify_3d(
    x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]
) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, h*w*d, p*q*r*c), where (p, q, r) is patch size,
            and H = h * p, W = w * q, D = d * r.
        patch_size: patch size (p, q, r).
        grid_size: grid size (h, w, d).

    Returns:
        image of shape (batch, in_chans, H, W, D)
    """
    batch = x.shape[0]
    p, q, r = patch_size
    h, w, d = grid_size
    x = x.reshape(shape=(batch, h, w, d, p, q, r, -1))
    x = torch.einsum("nhwdpqrc->nchpwqdr", x).contiguous()
    x = x.reshape(shape=(batch, -1, h * p, w * q, d * r))
    return x


def unpatchify_4d(
    x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]
) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, h*w*d*t, p*q*r*s*c), where (p, q, r, s) is patch size,
            and H = h * p, W = w * q, D = d * r, T = t * s.
        patch_size: patch size (p, q, r, s).
        grid_size: grid size (h, w, d, t).

    Returns:
        image of shape (batch, in_chans, H, W, D, T)
    """
    batch = x.shape[0]
    p, q, r, s = patch_size
    h, w, d, t = grid_size
    x = x.reshape(shape=(batch, h, w, d, t, p, q, r, s, -1))
    x = torch.einsum("nhwdtpqrsc->nchpwqdrts", x).contiguous()
    x = x.reshape(shape=(batch, -1, h * p, w * q, d * r, t * s))
    return x


def unpatchify(
    x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]
) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, n_patches, chans).
        patch_size: patch size.
        grid_size: grid size.

    Returns:
        image: (batch, in_chans, ...)
    """
    _, n_patches, chans = x.shape
    if n_patches != math.prod(grid_size):
        raise ValueError(
            f"Number of patches {n_patches} != product of grid size {math.prod(grid_size)} for {grid_size}."
        )
    if chans % math.prod(patch_size) != 0:
        raise ValueError(
            f"Number of channels {chans} is not divisible by product of patch size {math.prod(patch_size)} "
            f"for {patch_size}."
        )
    if len(patch_size) != len(grid_size):
        raise ValueError(
            f"Patch size {patch_size} and grid size {grid_size} do not match."
        )
    if len(patch_size) == 2:
        return unpatchify_2d(x, patch_size, grid_size)
    if len(patch_size) == 3:
        return unpatchify_3d(x, patch_size, grid_size)
    if len(patch_size) == 4:
        return unpatchify_4d(x, patch_size, grid_size)
    raise ValueError(
        f"Unpatchify only supports 2D, 3D, and 4D images, got {len(patch_size)}D."
    )
