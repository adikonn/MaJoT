#!/usr/bin/env python
"""Сравнение нейросети и классических бейзлайнов на одном тестовом наборе."""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.jacobi_type import joint_triangularize as jacobi_jt
from src.baseline.optim_newton import joint_triangularize as newton_jt
from src.baseline.pencil_schur import joint_triangularize as schur_jt
from src.evaluation.metrics import evaluate_transform
from src.models import build_model
from src.training.data import build_test_dataset


BASELINES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "jacobi": jacobi_jt,
    "newton": newton_jt,
    "schur": schur_jt,
}


def load_nn(config: dict, checkpoint: str, device: torch.device) -> Callable:
    model = build_model(config["model"]).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    def predict(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(A.to(device), B.to(device))

    return predict


def bench_one(
    name: str,
    predictor: Callable,
    test_data: list[dict[str, Any]],
    device: torch.device,
    warmup: int = 3,
) -> list[dict[str, Any]]:
    for sample in test_data[:warmup]:
        A, B = sample["A"].to(device), sample["B"].to(device)
        _ = predictor(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for sample in test_data:
        A = sample["A"].to(device)
        B = sample["B"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        T = predictor(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if not torch.is_tensor(T):
            T = torch.as_tensor(T, device=A.device, dtype=A.dtype)
        m = evaluate_transform(T, A, B)
        m.update(
            {
                "predictor": name,
                "n": sample["n"],
                "type": sample["type"],
                "time_seconds": elapsed,
            }
        )
        rows.append(m)
    return rows


def summarize(rows: list[dict[str, Any]]) -> None:
    by_pred: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_pred.setdefault(r["predictor"], []).append(r)

    print("\n=== Сводка (медиана по всем сэмплам) ===")
    for name, group in sorted(by_pred.items()):
        la = statistics.median(r["lower_ratio_A"] for r in group)
        lb = statistics.median(r["lower_ratio_B"] for r in group)
        orth = statistics.median(r["orth_residual"] for r in group)
        tmed = statistics.median(r["time_seconds"] for r in group)
        print(
            f"  {name:22s}  lower_A={la:.5f}  lower_B={lb:.5f}  "
            f"orth={orth:.5f}  time={tmed:.5f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Чекпойнт нейросети")
    parser.add_argument("--output", default="results/compare_all.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_data = build_test_dataset(config["data"])
    print(f"Устройство: {device}, тестовых сэмплов: {len(test_data)}")

    all_rows: list[dict[str, Any]] = []

    if args.checkpoint:
        nn_pred = load_nn(config, args.checkpoint, device)
        print("Бенчмарк нейросети...", flush=True)
        all_rows.extend(bench_one(config["model"]["name"], nn_pred, test_data, device))

    if not args.skip_baselines:
        for bname, fn in BASELINES.items():
            print(f"Бенчмарк {bname}...", flush=True)
            predictor = lambda A, B, f=fn: f(A.cpu(), B.cpu())
            all_rows.extend(bench_one(bname, predictor, test_data, device))

    if not all_rows:
        raise SystemExit("Нет результатов: укажите --checkpoint и/или не используйте --skip-baselines")

    if not args.skip_baselines:
        for bname, fn in BASELINES.items():
            print(f"Бенчмарк {bname}...", flush=True)
            predictor = lambda A, B, f=fn: f(A.cpu(), B.cpu())
            all_rows.extend(bench_one(bname, predictor, test_data, device))

    summarize(all_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
