#!/usr/bin/env python
"""Universal benchmark: evaluate any Triangularizer on the test set.

Supports neural network checkpoints out of the box. To benchmark a classical
baseline, write a small adapter (a callable `(A, B) -> T`) and pass it via
`--baseline <module>:<callable>`.

Examples:
    # Benchmark a trained neural network model:
    python scripts/benchmark.py \
        --config configs/matrix_transformer.yaml \
        --checkpoint checkpoints/matrix_transformer_v1/best_model.pt \
        --output results/matrix_transformer_v1.csv

    # Benchmark a baseline by import path (e.g., src.baseline.qz:solve):
    python scripts/benchmark.py \
        --config configs/matrix_transformer.yaml \
        --baseline src.baseline.qz:solve \
        --output results/qz.csv

"""
from __future__ import annotations

import argparse
import csv
import importlib
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_transform
from src.models import build_model
from src.training.data import build_test_dataset

if TYPE_CHECKING:
    from collections.abc import Callable


def load_model_predictor(config: dict, checkpoint_path: str, device: torch.device) -> tuple[Callable, str]:
    model = build_model(config["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    def predictor(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(A, B)

    name = config["model"]["name"]
    return predictor, name


def load_baseline_predictor(spec: str) -> tuple[Callable, str]:
    """Load `module:callable` and return it together with a display name."""
    if ":" not in spec:
        msg = f"Baseline spec must be 'module:callable', got {spec!r}"
        raise ValueError(msg)
    module_path, fn_name = spec.split(":")
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    return fn, f"{module_path}.{fn_name}"


def benchmark(
    predictor: Callable,
    test_data: list[dict[str, Any]],
    device: torch.device,
    warmup: int = 5,
    runs_per_sample: int = 3,
) -> list[dict[str, Any]]:
    for sample in test_data[:warmup]:
        A = sample["A"].to(device)
        B = sample["B"].to(device)
        _ = predictor(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for sample in test_data:
        A = sample["A"].to(device)
        B = sample["B"].to(device)

        times: list[float] = []
        T = None
        for _ in range(runs_per_sample):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            T = predictor(A, B)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        if not torch.is_tensor(T):
            T = torch.as_tensor(T, device=A.device, dtype=A.dtype)
        elif T.device != A.device:
            T = T.to(A.device)

        m = evaluate_transform(T, A, B)
        m["time_seconds"] = statistics.median(times)
        m["n"] = sample["n"]
        m["type"] = sample["type"]
        rows.append(m)
    return rows


NUMERIC_KEYS = ["lower_ratio_A", "lower_ratio_B", "orth_residual", "T_cond", "time_seconds"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Group rows by matrix type and compute mean/median per metric."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["type"], []).append(r)
    grouped["all"] = rows

    summary: dict[str, dict[str, float]] = {}
    for group_name, group_rows in grouped.items():
        s: dict[str, float] = {}
        for key in NUMERIC_KEYS:
            vals = [r[key] for r in group_rows if r[key] != float("inf")]
            if not vals:
                continue
            s[f"{key}_mean"] = float(sum(vals) / len(vals))
            s[f"{key}_median"] = float(statistics.median(vals))
        s["count"] = float(len(group_rows))
        summary[group_name] = s
    return summary


def print_summary(predictor_name: str, summary: dict[str, dict[str, float]]) -> None:
    for stats in summary.values():
        for k in stats:
            if k == "count":
                continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config used for the data + (optionally) model arch")
    parser.add_argument("--checkpoint", help="Path to a model checkpoint for NN predictors")
    parser.add_argument("--baseline", help="Baseline spec 'module:callable'")
    parser.add_argument("--output", default=None, help="Optional CSV path for per-sample results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    if not args.checkpoint and not args.baseline:
        parser.error("Provide either --checkpoint (for an NN) or --baseline (for a classical method)")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        predictor, name = load_model_predictor(config, args.checkpoint, device)
    else:
        predictor, name = load_baseline_predictor(args.baseline)

    test_data = build_test_dataset(config["data"])

    rows = benchmark(predictor, test_data, device=device)
    summary = summarize(rows)
    print_summary(name, summary)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if args.wandb:
        wandb.init(
            project=args.wandb_project or config["wandb"]["project"],
            entity=args.wandb_entity or config["wandb"].get("entity"),
            job_type="benchmark",
            name=f"bench/{name}",
            config={"predictor": name, "config_file": args.config},
        )
        for group_name, stats in summary.items():
            for k, v in stats.items():
                wandb.run.summary[f"{group_name}/{k}"] = v
        wandb.finish()


if __name__ == "__main__":
    main()
