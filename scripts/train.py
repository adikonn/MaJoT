#!/usr/bin/env python
"""Universal training entry point.

Usage:
    python scripts/train.py --config configs/matrix_transformer.yaml
    python scripts/train.py --config configs/matrix_transformer.yaml \
        --override training.lr=3e-4 model.num_layers=6 wandb.mode=offline
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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

    set_seed(config["seed"])

    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_ds, val_ds = build_datasets(config["data"])
    print(f"Train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_sampler=GroupedByNBatchSampler(
            train_ds, config["data"]["batch_size"], shuffle=True, seed=config["seed"]
        ),
        collate_fn=collate_by_n,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=GroupedByNBatchSampler(
            val_ds, config["data"]["batch_size"], shuffle=False
        ),
        collate_fn=collate_by_n,
    )

    model = build_model(config["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['name']} | {n_params:,} parameters")

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        mode=config["wandb"].get("mode", "online"),
        name=config["experiment_name"],
        tags=config["wandb"].get("tags"),
        config=config,
    )
    wandb.run.summary["n_params"] = n_params

    checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{config['experiment_name']}"))
    train(model, train_loader, val_loader, config, device, checkpoint_dir)

    wandb.finish()


if __name__ == "__main__":
    main()