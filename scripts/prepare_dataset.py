#!/usr/bin/env python
"""Предгенерация кэша синтетического датасета (запускать на login-ноде до GPU-задачи)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.data import build_datasets, build_test_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать и закэшировать train/val/test")
    parser.add_argument("--config", required=True, help="YAML-конфиг эксперимента")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    cache_dir = Path(data_cfg.get("cache_dir", "dataset/cache"))
    print(f"Кэш: {cache_dir.resolve()}", flush=True)

    t0 = time.perf_counter()
    train_ds, val_ds = build_datasets(data_cfg)
    print(
        f"Train/val: {len(train_ds)} / {len(val_ds)} сэмплов за {time.time() - t0:.1f} с",
        flush=True,
    )

    t0 = time.perf_counter()
    test = build_test_dataset(data_cfg)
    print(f"Test: {len(test)} сэмплов за {time.time() - t0:.1f} с", flush=True)
    print("Готово.", flush=True)


if __name__ == "__main__":
    main()
