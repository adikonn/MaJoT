#!/usr/bin/env python
"""Точка входа для обучения моделей MaJoT.

Примеры:
    python scripts/train.py --config configs/dual_stream_rowcol_hpc.yaml
    python scripts/train.py --config configs/dual_stream_rowcol.yaml \\
        --override training.lr=3e-4 wandb.mode=offline
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

# Make the project root importable regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model  # noqa: E402
from src.training.data import (  # noqa: E402
    GroupedByNBatchSampler,
    build_datasets,
    collate_by_n,
)
from src.training.trainer import train  # noqa: E402


def set_seed(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply a list of `dotted.key=value` overrides to a nested config dict."""
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Override '{kv}' is not in key=value form")
        key, raw_value = kv.split("=", 1)
        # Try to parse the value as YAML (so true/false/numbers/lists come out typed).
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d:
                raise KeyError(f"Override key {key} hits missing intermediate '{k}'")
            d = d[k]
        d[keys[-1]] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config keys, e.g. training.lr=1e-4",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = apply_overrides(config, args.override)

    set_seed(config["seed"], deterministic=bool(config.get("deterministic", False)))

    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}", flush=True)

    train_ds, val_ds = build_datasets(config["data"])
    print(f"Train samples: {len(train_ds)}, val samples: {len(val_ds)}", flush=True)

    data_cfg = config["data"]
    loader_kw: dict = {
        "collate_fn": collate_by_n,
        "num_workers": int(data_cfg.get("num_workers", 0)),
    }
    if loader_kw["num_workers"] > 0:
        loader_kw["persistent_workers"] = True
    if data_cfg.get("pin_memory") and device.type == "cuda":
        loader_kw["pin_memory"] = True

    train_loader = DataLoader(
        train_ds,
        batch_sampler=GroupedByNBatchSampler(
            train_ds, data_cfg["batch_size"], shuffle=True, seed=config["seed"]
        ),
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=GroupedByNBatchSampler(
            val_ds, data_cfg["batch_size"], shuffle=False
        ),
        **loader_kw,
    )

    model = build_model(config["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['name']} | {n_params:,} parameters", flush=True)

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        mode=config["wandb"].get("mode", "online"),
        name=config["experiment_name"],
        tags=config["wandb"].get("tags"),
        config=config,
    )
    wandb.run.summary["n_params"] = n_params

    checkpoint_dir = Path(
        config.get("checkpoint_dir", f"checkpoints/{config['experiment_name']}")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    train(model, train_loader, val_loader, config, device, checkpoint_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
