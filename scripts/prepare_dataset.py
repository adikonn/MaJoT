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

def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать и закэшировать train/val/test")
    parser.add_argument("--config", required=True, help="YAML-конфиг эксперимента")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    Path(data_cfg.get("cache_dir", "dataset/cache"))

    time.perf_counter()
    _train_ds, _val_ds = build_datasets(data_cfg)

    time.perf_counter()
    build_test_dataset(data_cfg)


if __name__ == "__main__":
    main()
