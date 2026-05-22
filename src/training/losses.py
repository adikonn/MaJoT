"""Loss functions for joint triangularization."""
from __future__ import annotations

import torch


def triangularization_loss(T: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Mean Frobenius norm squared of the strictly lower-triangular part of T^T A T and T^T B T.

    Shapes:
        T, A, B: (batch, n, n) or (n, n)
    """
    Ap = T.transpose(-1, -2) @ A @ T
    Bp = T.transpose(-1, -2) @ B @ T
    sub_a = torch.tril(Ap, diagonal=-1)
    sub_b = torch.tril(Bp, diagonal=-1)
    loss_a = sub_a.pow(2).sum(dim=(-1, -2))
    loss_b = sub_b.pow(2).sum(dim=(-1, -2))
    return (loss_a + loss_b).mean()


def orthogonality_loss(T: torch.Tensor) -> torch.Tensor:
    """Penalty pulling T toward an orthogonal matrix: || T^T T - I ||_F^2 (mean over batch)."""
    n = T.shape[-1]
    eye = torch.eye(n, device=T.device, dtype=T.dtype)
    err = T.transpose(-1, -2) @ T - eye
    return err.pow(2).sum(dim=(-1, -2)).mean()


def total_loss(
    T: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_orth: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Aggregate objective. Returns (loss, dict of component values for logging)."""
    l_tri = triangularization_loss(T, A, B)
    l_orth = orthogonality_loss(T)
    loss = l_tri + lambda_orth * l_orth
    return loss, {
        "loss_tri": float(l_tri.detach()),
        "loss_orth": float(l_orth.detach()),
    }
