"""Module for patch masking."""

from __future__ import annotations

import torch
from torch.distributions.dirichlet import Dirichlet


def get_batch_random_patch_mask(
    batch_size: int, n_patches: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Get a per-sample random mask for a tensor of shape (batch, n_patches, emb_dim).

    Per-sample shuffling is done by argsort random noise.

    https://github.com/EPFL-VILAB/MultiMAE/blob/66910f5b5ba236f5e731883db85fe4f24ee01106/multimae/multimae.py#L164

    Args:
        batch_size: batch size.
        n_patches: number of patches in total.
        mask_ratio: ratio of patches to remove, in [0, 1]. For each sample in the batch, ratio is the same.
        device: device to store the results.

    Returns:
        mask: binary mask, (batch, n_patches), 0 is keep, 1 is remove.
    """
    if mask_ratio < 0:
        raise ValueError(f"mask_ratio must be positive, got {mask_ratio}.")
    if mask_ratio == 0:
        return torch.zeros((batch_size, n_patches), dtype=torch.bool, device=device)

    # sort noise for each sample
    noise = torch.rand(batch_size, n_patches, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep, 1 is remove
    n_keep = int(n_patches * (1 - mask_ratio))
    mask = torch.ones([batch_size, n_patches], device=device, dtype=torch.bool)
    mask[:, :n_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask
