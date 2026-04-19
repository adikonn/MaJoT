import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List


def _run_command(args: List[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args)}")


def _read_metrics_md(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return path.read_text(encoding="utf-8")


def _extract_row_value(md: str, prefix: str) -> float:
    for line in md.splitlines():
        if line.startswith(prefix):
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                return float(match.group(1))
    raise ValueError(f"Could not find row '{prefix}' in metrics report.")


def _extract_algorithm_metric(md: str, algo_prefix: str, column_idx: int) -> float:
    # columns: [name, accuracy, precision, recall, f1]
    for line in md.splitlines():
        if line.startswith(algo_prefix):
            parts = [p.strip() for p in line.split("|")]
            value = parts[column_idx]
            cleaned = value.replace("`", "").replace("*", "").strip()
            return float(cleaned)
    raise ValueError(f"Could not find algorithm row '{algo_prefix}'")


def run_experiment(
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    reg_identity: float,
    class_margin_weight: float,
    class_margin: float,
    calib_ratio: float,
) -> Dict[str, float]:
    weights = f"weights_seed_{seed}.pt"
    history = f"history_seed_{seed}.json"

    _run_command(
        [
            "python",
            "train.py",
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--patience",
            str(patience),
            "--reg-identity",
            str(reg_identity),
            "--class-margin-weight",
            str(class_margin_weight),
            "--class-margin",
            str(class_margin),
            "--weights",
            weights,
            "--history",
            history,
        ]
    )
    _run_command(
        [
            "python",
            "evaluate.py",
            "--weights",
            weights,
            "--batch-size",
            str(batch_size),
            "--calib-ratio",
            str(calib_ratio),
            "--seed",
            str(seed),
        ]
    )

    md = _read_metrics_md(Path("metrics.md"))
    return {
        "seed": float(seed),
        "general_score": _extract_row_value(md, "| **Итоговая ошибка (General Score)**"),
        "triang_score": _extract_row_value(md, "| **Ошибка на трианг. матрицах (Triang Score)**"),
        "nn_accuracy": _extract_algorithm_metric(md, "| NN (по score <= threshold)", 2),
        "nn_f1": _extract_algorithm_metric(md, "| NN (по score <= threshold)", 5),
        "classical_accuracy": _extract_algorithm_metric(md, "| Classical criterion", 2),
        "classical_f1": _extract_algorithm_metric(md, "| Classical criterion", 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed experiments and aggregate metrics.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 42], help="List of random seeds.")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for train/eval.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--reg-identity", type=float, default=1e-3, help="Identity regularization coefficient.")
    parser.add_argument("--class-margin-weight", type=float, default=0.25, help="Weight of class margin term.")
    parser.add_argument("--class-margin", type=float, default=0.25, help="Class separation margin value.")
    parser.add_argument("--calib-ratio", type=float, default=0.25, help="Calibration split ratio for threshold.")
    parser.add_argument(
        "--report",
        type=str,
        default="experiments_report.md",
        help="Path to aggregated markdown report.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="experiments_results.json",
        help="Path to raw json results.",
    )
    args = parser.parse_args()

    results: List[Dict[str, float]] = []
    for seed in args.seeds:
        print(f"[exp] running seed={seed}")
        results.append(
            run_experiment(
                seed,
                args.epochs,
                args.batch_size,
                args.lr,
                args.patience,
                args.reg_identity,
                args.class_margin_weight,
                args.class_margin,
                args.calib_ratio,
            )
        )

    agg: Dict[str, float] = {}
    for key in [
        "general_score",
        "triang_score",
        "nn_accuracy",
        "nn_f1",
        "classical_accuracy",
        "classical_f1",
    ]:
        values = [r[key] for r in results]
        agg[f"{key}_mean"] = sum(values) / len(values)
        agg[f"{key}_min"] = min(values)
        agg[f"{key}_max"] = max(values)

    Path(args.json).write_text(
        json.dumps({"runs": results, "aggregate": agg}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Multi-seed Experiment Report",
        "",
        "## Runs",
        "",
        "| seed | general_score | triang_score | nn_acc | nn_f1 | classical_acc | classical_f1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| {int(row['seed'])} | {row['general_score']:.6f} | {row['triang_score']:.6f} | "
            f"{row['nn_accuracy']:.6f} | {row['nn_f1']:.6f} | "
            f"{row['classical_accuracy']:.6f} | {row['classical_f1']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| metric | mean | min | max |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in [
        "general_score",
        "triang_score",
        "nn_accuracy",
        "nn_f1",
        "classical_accuracy",
        "classical_f1",
    ]:
        lines.append(
            f"| {key} | {agg[f'{key}_mean']:.6f} | {agg[f'{key}_min']:.6f} | {agg[f'{key}_max']:.6f} |"
        )

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[exp] report saved to {args.report}")
    print(f"[exp] json saved to {args.json}")


if __name__ == "__main__":
    main()
