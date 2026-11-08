from __future__ import annotations

import itertools
from typing import Tuple

import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est.float().flatten()
    ref = ref.float().flatten()

    length = min(est.numel(), ref.numel())
    est = est[:length]
    ref = ref[:length]

    est = est - est.mean()
    ref = ref - ref.mean()

    ref_energy = torch.sum(ref * ref) + eps
    target = torch.sum(est * ref) * ref / ref_energy
    noise = est - target

    return 10.0 * torch.log10((torch.sum(target * target) + eps) / (torch.sum(noise * noise) + eps))


def pit_match(
    estimates: torch.Tensor,
    references: torch.Tensor,
    return_info: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Tuple[int, ...], float, float]:
    """Permutation-invariant matching using mean SI-SDR.

    Parameters
    ----------
    estimates:
        Tensor shaped (N, T). Speaker order is arbitrary.
    references:
        Tensor shaped (N, T). Speaker order is ground truth order.
    return_info:
        If True, also returns permutation, summed SI-SDR, and mean SI-SDR.

    Returns
    -------
    reordered_estimates:
        Tensor shaped (N, T), ordered to match references.
    perm:
        Tuple where perm[ref_idx] is the chosen estimate index for that reference.
    """
    if estimates.ndim != 2 or references.ndim != 2:
        raise ValueError(
            f"PIT expects (N,T) tensors, got estimates={tuple(estimates.shape)}, references={tuple(references.shape)}"
        )
    if estimates.shape[0] != references.shape[0]:
        raise ValueError(
            f"PIT speaker count mismatch: estimates={estimates.shape[0]}, references={references.shape[0]}"
        )

    estimates = estimates.detach().cpu().float()
    references = references.detach().cpu().float()

    n = int(references.shape[0])
    min_len = min(int(estimates.shape[-1]), int(references.shape[-1]))
    estimates = estimates[:, :min_len]
    references = references[:, :min_len]

    best_perm: Tuple[int, ...] | None = None
    best_score = -float("inf")

    # perm[ref_idx] = estimate index assigned to that reference.
    for perm in itertools.permutations(range(n)):
        score = 0.0
        for ref_idx, est_idx in enumerate(perm):
            score += float(si_sdr(estimates[est_idx], references[ref_idx]))
        if score > best_score:
            best_score = score
            best_perm = tuple(perm)

    if best_perm is None:
        raise RuntimeError("PIT failed to find a valid permutation.")

    reordered = torch.stack([estimates[est_idx] for est_idx in best_perm], dim=0).contiguous()
    mean_score = best_score / n

    if return_info:
        return reordered, best_perm, float(best_score), float(mean_score)
    return reordered
