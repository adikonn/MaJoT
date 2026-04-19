import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Failed command: {' '.join(cmd)}")


def _load_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_config(config: Dict, seeds: List[int], n_joint: int, n_random: int, n_size: int) -> Dict:
    name = config["name"]
    print(f"[auto] running config: {name}")

    gen_cmd = [
        "python",
        "dataset/generate_data.py",
        "--n-joint",
        str(n_joint),
        "--n-random",
        str(n_random),
        "--n-size",
        str(n_size),
        "--seed",
        str(config["dataset_seed"]),
    ]
    if config["allow_noisy_negatives"]:
        gen_cmd.append("--allow-noisy-negatives")
    _run(gen_cmd)

    json_out = f"results_{name}.json"
    report_out = f"report_{name}.md"
    exp_cmd = [
        "python",
        "run_experiments.py",
        "--seeds",
        *[str(s) for s in seeds],
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--patience",
        str(config["patience"]),
        "--reg-identity",
        str(config["reg_identity"]),
        "--class-margin-weight",
        str(config["class_margin_weight"]),
        "--class-margin",
        str(config["class_margin"]),
        "--calib-ratio",
        str(config["calib_ratio"]),
        "--report",
        report_out,
        "--json",
        json_out,
    ]
    _run(exp_cmd)
    data = _load_json(json_out)
    return {
        "name": name,
        "config": config,
        "aggregate": data["aggregate"],
        "raw_json": json_out,
        "report_md": report_out,
    }


def _score(entry: Dict) -> float:
    agg = entry["aggregate"]
    # Weighted objective: higher f1/acc is good, lower general score is good.
    return (
        0.60 * agg["nn_f1_mean"]
        + 0.35 * agg["nn_accuracy_mean"]
        - 0.05 * min(1.0, agg["general_score_mean"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full auto research pipeline with ablations and sweep.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 42])
    parser.add_argument("--n-joint", type=int, default=400)
    parser.add_argument("--n-random", type=int, default=400)
    parser.add_argument("--n-size", type=int, default=3)
    parser.add_argument("--output-report", default="auto_research_report.md")
    parser.add_argument("--output-json", default="auto_research_results.json")
    args = parser.parse_args()

    configs: List[Dict] = [
        {
            "name": "baseline_no_margin_noisy_neg",
            "dataset_seed": 42,
            "allow_noisy_negatives": True,
            "epochs": 40,
            "batch_size": 64,
            "lr": 0.002,
            "patience": 10,
            "reg_identity": 1e-3,
            "class_margin_weight": 0.0,
            "class_margin": 0.0,
            "calib_ratio": 0.25,
        },
        {
            "name": "hard_neg_margin_mid",
            "dataset_seed": 42,
            "allow_noisy_negatives": False,
            "epochs": 50,
            "batch_size": 64,
            "lr": 0.0015,
            "patience": 12,
            "reg_identity": 1e-3,
            "class_margin_weight": 0.25,
            "class_margin": 0.25,
            "calib_ratio": 0.25,
        },
        {
            "name": "hard_neg_margin_strong",
            "dataset_seed": 42,
            "allow_noisy_negatives": False,
            "epochs": 60,
            "batch_size": 64,
            "lr": 0.0012,
            "patience": 15,
            "reg_identity": 1e-3,
            "class_margin_weight": 0.35,
            "class_margin": 0.40,
            "calib_ratio": 0.25,
        },
    ]

    results = [run_config(cfg, args.seeds, args.n_joint, args.n_random, args.n_size) for cfg in configs]
    for item in results:
        item["selection_score"] = _score(item)
    results.sort(key=lambda x: x["selection_score"], reverse=True)
    best = results[0]

    Path(args.output_json).write_text(
        json.dumps({"results": results, "best": best}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Auto Research Report",
        "",
        "## Ranked Configurations",
        "",
        "| rank | config | nn_acc_mean | nn_f1_mean | general_score_mean | selection_score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, item in enumerate(results, start=1):
        agg = item["aggregate"]
        lines.append(
            f"| {i} | {item['name']} | {agg['nn_accuracy_mean']:.6f} | {agg['nn_f1_mean']:.6f} | "
            f"{agg['general_score_mean']:.6f} | {item['selection_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Best Configuration",
            "",
            f"- name: `{best['name']}`",
            f"- nn_acc_mean: `{best['aggregate']['nn_accuracy_mean']:.6f}`",
            f"- nn_f1_mean: `{best['aggregate']['nn_f1_mean']:.6f}`",
            f"- general_score_mean: `{best['aggregate']['general_score_mean']:.6f}`",
            f"- source report: `{best['report_md']}`",
            f"- source json: `{best['raw_json']}`",
        ]
    )
    Path(args.output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[auto] saved report: {args.output_report}")
    print(f"[auto] saved json: {args.output_json}")


if __name__ == "__main__":
    main()
