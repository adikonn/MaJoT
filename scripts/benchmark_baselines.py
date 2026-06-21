import contextlib
import os
import time
from typing import cast

import pandas as pd
import torch
from tqdm import tqdm

from baseline.jacobi_type import joint_triangularize as jacobi_jt
from baseline.optim_newton import joint_triangularize as newton_jt
from baseline.pencil_schur import joint_triangularize as schur_jt

BASELINES = {"Schur": schur_jt, "Jacobi": jacobi_jt, "Newton": newton_jt}


def compute_rel_residual(A, B, Q):
    """Вычисляет относительный нижнетреугольный residual после применения преобразования Q."""
    A_prime = Q.T @ A @ Q
    B_prime = Q.T @ B @ Q

    def tril_sq_sum(M):
        return torch.sum(torch.tril(M, diagonal=-1) ** 2)

    num = tril_sq_sum(A_prime) + tril_sq_sum(B_prime)
    den = torch.sum(A**2) + torch.sum(B**2) + 1e-12
    return (num / den).item()


def run_benchmark() -> None:
    dataset_path = "dataset/dataset.pt"
    if not os.path.exists(dataset_path):
        return

    dataset = torch.load(dataset_path)

    results = []

    for sample in tqdm(dataset, desc="Benchmarking"):
        n = sample["n"]
        mtype = sample["type"]
        A = sample["A"]
        B = sample["B"]

        for alg_name, alg_fn in BASELINES.items():
            try:
                start_time = time.perf_counter()
                Q = alg_fn(A, B)
                end_time = time.perf_counter()

                elapsed = end_time - start_time
                residual = compute_rel_residual(A, B, Q)

                results.append(
                    {
                        "Algorithm": alg_name,
                        "Type": mtype,
                        "n": n,
                        "Time (s)": elapsed,
                        "Rel_Residual": residual,
                    },
                )
            except Exception:
                results.append(
                    {
                        "Algorithm": alg_name,
                        "Type": mtype,
                        "n": n,
                        "Time (s)": float("nan"),
                        "Rel_Residual": float("nan"),
                    },
                )

    df = pd.DataFrame(results)

    agg_df = cast(
        "pd.DataFrame",
        df.groupby(["Algorithm", "Type", "n"], as_index=False).agg(
            {"Time (s)": "mean", "Rel_Residual": "mean"},
        ),
    )

    agg_df = agg_df.sort_values(by=["Type", "n", "Algorithm"])


    with contextlib.suppress(ImportError):
        pass


if __name__ == "__main__":
    run_benchmark()
