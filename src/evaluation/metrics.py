"""Geometric / numerical metrics for evaluating a transformation T."""
from __future__ import annotations

import torch


def lower_norm_ratio(M: torch.Tensor) -> torch.Tensor:
    """Strictly-lower-triangular Frobenius norm divided by the total Frobenius norm.

    Returns 0 if M is itself essentially zero. Smaller is better; 0 means M is
    already upper triangular.
    """
    total = M.norm()
    if float(total) < 1e-12:
        return torch.zeros((), device=M.device, dtype=M.dtype)
    return torch.tril(M, diagonal=-1).norm() / total


def evaluate_transform(T: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> dict[str, float]:
    """Compute a battery of evaluation metrics for a single (T, A, B) triple."""
    Ap = T.transpose(-1, -2) @ A @ T
    Bp = T.transpose(-1, -2) @ B @ T

    metrics = {
        "lower_ratio_A": float(lower_norm_ratio(Ap)),
        "lower_ratio_B": float(lower_norm_ratio(Bp)),
        "lower_norm_A": float(torch.tril(Ap, diagonal=-1).norm()),
        "lower_norm_B": float(torch.tril(Bp, diagonal=-1).norm()),
    }
    # Conditioning of T: how far we are from singular.
    try:
        metrics["T_cond"] = float(torch.linalg.cond(T))
    except Exception:
        metrics["T_cond"] = float("inf")
    # Orthogonality residual.
    n = T.shape[-1]
    eye = torch.eye(n, device=T.device, dtype=T.dtype)
    metrics["orth_residual"] = float((T.transpose(-1, -2) @ T - eye).norm())
    return metrics
