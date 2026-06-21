#!/usr/bin/env python
"""Гибрид: мульти-старт сети (тёплый старт) -> несколько свипов Jacobi.

Сравниваем на одном тест-сете:
  - NN multi-start (K) в одиночку;
  - Jacobi с нуля до сходимости (референс качества/скорости бейзлайна);
  - Hybrid: T0 = multi-start(K); A'=T0^T A T0; Q_j = Jacobi(A',B', max_sweeps=S);
            T = T0 @ Q_j   — для нескольких малых S.

Метрика — lower_ratio (как в evaluation/metrics), время — мс/сэмпл.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


from src.baseline.jacobi_type import joint_triangularize
from src.dataset.generate_data import generate_noisy, generate_perfect
from src.evaluation.metrics import lower_norm_ratio
from src.models import build_model


def lr_pair(T, A, B):
    Ap = T.transpose(-1, -2) @ A @ T
    Bp = T.transpose(-1, -2) @ B @ T
    return 0.5 * (lower_norm_ratio(Ap) + lower_norm_ratio(Bp))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        "checkpoints/matrix_transformer_ortho_prod_v1/best_model.pt",
        map_location=device, weights_only=False,
    )
    model = build_model(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    K = 8           # рестартов в тёплом старте
    sweeps = [1, 2, 3]
    ns = [4, 8, 12, 16]
    per_n = 2

    torch.manual_seed(7)
    for _typ, gen in [("perfect", generate_perfect),
                     ("noisy", lambda n: generate_noisy(n, noise_level=1e-3))]:
        ["multistart"] + [f"hybrid S={s}" for s in sweeps] + ["jacobi(full)"]
        for n in ns:
            pairs = [gen(n) for _ in range(per_n)]
            A = torch.stack([p[0] for p in pairs]).to(device)
            B = torch.stack([p[1] for p in pairs]).to(device)

            if device.type == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            best_lr = None; T0 = None
            for k in range(K):
                if k == 0:
                    R = torch.eye(n, device=device).expand(A.shape[0], n, n)
                else:
                    H = torch.randn(A.shape[0], n, n, device=device)
                    Q, Rr = torch.linalg.qr(H)
                    R = Q * torch.sign(torch.diagonal(Rr, dim1=-2, dim2=-1)).unsqueeze(1)
                Tp = model(R.transpose(-1, -2) @ A @ R, R.transpose(-1, -2) @ B @ R)
                Tc = R @ Tp
                lr = torch.stack([lr_pair(Tc[i], A[i], B[i]) for i in range(A.shape[0])])
                if best_lr is None:
                    best_lr, T0 = lr, Tc
                else:
                    take = lr < best_lr
                    best_lr = torch.where(take, lr, best_lr)
                    T0 = torch.where(take.view(-1, 1, 1), Tc, T0)
            if device.type == "cuda": torch.cuda.synchronize()
            ms_ms = (time.perf_counter() - t) / A.shape[0] * 1000
            lr_ms = best_lr.mean().item()

            results = [(lr_ms, ms_ms)]

            Acpu, Bcpu, T0cpu = A.cpu(), B.cpu(), T0.cpu()
            for S in sweeps:
                t = time.perf_counter()
                lrs = []
                for i in range(A.shape[0]):
                    Ap = T0cpu[i].T @ Acpu[i] @ T0cpu[i]
                    Bp = T0cpu[i].T @ Bcpu[i] @ T0cpu[i]
                    Qj = joint_triangularize(Ap, Bp, max_sweeps=S)
                    T = T0cpu[i] @ Qj
                    lrs.append(lr_pair(T, Acpu[i], Bcpu[i]).item())
                hy_ms = (time.perf_counter() - t) / A.shape[0] * 1000 + ms_ms
                results.append((sum(lrs) / len(lrs), hy_ms))

            t = time.perf_counter()
            lrs = []
            for i in range(A.shape[0]):
                Qj = joint_triangularize(Acpu[i], Bcpu[i])
                lrs.append(lr_pair(Qj, Acpu[i], Bcpu[i]).item())
            jac_ms = (time.perf_counter() - t) / A.shape[0] * 1000
            results.append((sum(lrs) / len(lrs), jac_ms))

            "".join(f"{lr:.3f}|{ms:6.1f}ms ".ljust(16) for lr, ms in results)


if __name__ == "__main__":
    main()
