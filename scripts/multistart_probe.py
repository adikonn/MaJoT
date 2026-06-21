#!/usr/bin/env python
"""Прототип test-time мульти-старта для matrix_transformer_ortho.

Идея: сеть детерминирована, но мы диверсифицируем вход случайной ортогональной
заменой базиса. Для кандидата k берём случайную R_k и считаем
    T_k = R_k @ net(R_k^T A R_k, R_k^T B R_k),
что тоже триангуляризует исходную (A,B). Из K кандидатов берём лучший по lower_ratio.
Сравниваем с одношаговым (K=1, R=I) инференсом.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.generate_data import generate_noisy, generate_perfect
from src.evaluation.metrics import lower_norm_ratio
from src.models import build_model


def random_orthogonal(batch: int, n: int, device) -> torch.Tensor:
    H = torch.randn(batch, n, n, device=device)
    Q, R = torch.linalg.qr(H)
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    return Q * sign.unsqueeze(1)


@torch.no_grad()
def lower_ratio_mean(T, A, B):
    """Средний (A,B) lower_ratio покомпонентно -> (batch,)."""
    Ap = T.transpose(-1, -2) @ A @ T
    Bp = T.transpose(-1, -2) @ B @ T
    ra = torch.stack([lower_norm_ratio(Ap[i]) for i in range(Ap.shape[0])])
    rb = torch.stack([lower_norm_ratio(Bp[i]) for i in range(Bp.shape[0])])
    return 0.5 * (ra + rb)


@torch.no_grad()
def multistart(model, A, B, K, device):
    """Возвращает best-of-K lower_ratio на сэмпл (batch,)."""
    batch, n, _ = A.shape
    best = None
    for k in range(K):
        if k == 0:
            R = torch.eye(n, device=device).expand(batch, n, n)
        else:
            R = random_orthogonal(batch, n, device)
        Ar = R.transpose(-1, -2) @ A @ R
        Br = R.transpose(-1, -2) @ B @ R
        Tp = model(Ar, Br)
        T = R @ Tp
        lr = lower_ratio_mean(T, A, B)
        best = lr if best is None else torch.minimum(best, lr)
    return best


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        "checkpoints/matrix_transformer_ortho_prod_v1/best_model.pt",
        map_location=device, weights_only=False,
    )
    model = build_model(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ns = [4, 6, 8, 12, 16]
    per_n = 200
    Ks = [1, 4, 8, 16, 32]

    torch.manual_seed(2024)
    for _typ, gen in [("perfect", generate_perfect),
                     ("noisy", lambda n: generate_noisy(n, noise_level=1e-3))]:
        "  n   " + "".join(f"K={k:<7}" for k in Ks)
        for n in ns:
            pairs = [gen(n) for _ in range(per_n)]
            A = torch.stack([p[0] for p in pairs]).to(device)
            B = torch.stack([p[1] for p in pairs]).to(device)
            row = f"  {n:<4}"
            for K in Ks:
                m = multistart(model, A, B, K, device).mean().item()
                row += f"{m:<9.4f}"


if __name__ == "__main__":
    main()
