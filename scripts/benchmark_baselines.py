#!/usr/bin/env python
"""Бенчмарк классических бейзлайнов на том же тестовом наборе, что и нейросеть."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_predictors import bench_one, summarize  # noqa: E402
from src.baseline.jacobi_type import joint_triangularize as jacobi_jt  # noqa: E402
from src.baseline.optim_newton import joint_triangularize as newton_jt  # noqa: E402
from src.baseline.pencil_schur import joint_triangularize as schur_jt  # noqa: E402
from src.training.data import build_test_dataset  # noqa: E402

BASELINES = {
    "schur": schur_jt,
    "jacobi": jacobi_jt,
    "newton": newton_jt,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dual_stream_rowcol_hpc.yaml")
    parser.add_argument("--output", default="results/benchmark_baselines.csv")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_data = build_test_dataset(config["data"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows: list[dict] = []
    for name, fn in BASELINES.items():
        print(f"Бенчмарк {name}...", flush=True)
        predictor = lambda A, B, f=fn: f(A.cpu(), B.cpu())
        all_rows.extend(bench_one(name, predictor, test_data, device))

    summarize(all_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV: {out}", flush=True)


if __name__ == "__main__":
    main()
