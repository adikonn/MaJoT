"""Обёртки датасета и сэмплер батчей с группировкой по размеру n.

Используем генераторы из `src.dataset.generate_data`. Так как n меняется между
сэмплами, произвольные элементы нельзя сложить в один тензор — `GroupedByNBatchSampler`
отдаёт батчи только с одинаковым n.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

# These imports point to the existing dataset module the project already has.
from src.dataset.generate_data import generate_perfect, generate_noisy, generate_random


_GENERATORS = {
    "perfect": lambda n, **kw: generate_perfect(n),
    "noisy": lambda n, noise_level=1e-3, **kw: generate_noisy(n, noise_level=noise_level),
    "random": lambda n, **kw: generate_random(n),
}


class MatrixDataset(Dataset):
    """A flat list of {n, type, A, B} dicts."""

    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def _cache_key(data_cfg: dict[str, Any], split: str) -> str:
    payload = {
        "split": split,
        "ns": data_cfg["ns"],
        "types": data_cfg["types"],
        "samples_per_n_per_type": data_cfg.get("samples_per_n_per_type"),
        "test_samples_per_n_per_type": data_cfg.get("test_samples_per_n_per_type"),
        "noise_level": data_cfg.get("noise_level", 1e-3),
        "seed": data_cfg.get("seed", 42),
        "test_seed": data_cfg.get("test_seed", 999),
        "val_fraction": data_cfg.get("val_fraction", 0.1),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cached_samples(cache_path: Path) -> list[dict[str, Any]] | None:
    if not cache_path.is_file():
        return None
    try:
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(cache_path, map_location="cpu")
    return blob["samples"]


def _save_cached_samples(cache_path: Path, samples: list[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples}, cache_path)


def _generate_samples(
    ns: list[int],
    types: list[str],
    samples_per_n_per_type: int,
    noise_level: float,
    seed: int,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    if cache_path is not None:
        cached = _load_cached_samples(cache_path)
        if cached is not None:
            return cached

    torch.manual_seed(seed)
    samples: list[dict[str, Any]] = []
    for n in ns:
        for t in types:
            gen = _GENERATORS[t]
            for _ in range(samples_per_n_per_type):
                A, B = gen(n, noise_level=noise_level)
                samples.append({"n": n, "type": t, "A": A, "B": B})

    if cache_path is not None:
        _save_cached_samples(cache_path, samples)
    return samples


def build_datasets(data_cfg: dict[str, Any]) -> tuple[MatrixDataset, MatrixDataset]:
    """Собрать train/val из конфига (с опциональным дисковым кэшем)."""
    cache_dir = Path(data_cfg.get("cache_dir", "dataset/cache"))
    cache_path = cache_dir / f"train_{_cache_key(data_cfg, 'train')}.pt"
    samples = _generate_samples(
        ns=data_cfg["ns"],
        types=data_cfg["types"],
        samples_per_n_per_type=data_cfg["samples_per_n_per_type"],
        noise_level=data_cfg.get("noise_level", 1e-3),
        seed=data_cfg.get("seed", 42),
        cache_path=cache_path,
    )
    rng = random.Random(data_cfg.get("seed", 42))
    rng.shuffle(samples)

    val_size = int(len(samples) * data_cfg["val_fraction"])
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    return MatrixDataset(train_samples), MatrixDataset(val_samples)


def build_test_dataset(data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Детерминированный тестовый набор для бенчмарка."""
    cache_dir = Path(data_cfg.get("cache_dir", "dataset/cache"))
    cache_path = cache_dir / f"test_{_cache_key(data_cfg, 'test')}.pt"
    return _generate_samples(
        ns=data_cfg["ns"],
        types=data_cfg["types"],
        samples_per_n_per_type=data_cfg.get("test_samples_per_n_per_type", 50),
        noise_level=data_cfg.get("noise_level", 1e-3),
        seed=data_cfg.get("test_seed", 999),
        cache_path=cache_path,
    )


class GroupedByNBatchSampler:
    """Yields batches (lists of indices) such that all items in a batch share the same n."""

    def __init__(
        self,
        dataset: MatrixDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)

        self._groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            self._groups[dataset[i]["n"]].append(i)

    def __iter__(self) -> Iterator[list[int]]:
        all_batches: list[list[int]] = []
        for indices in self._groups.values():
            indices = list(indices)
            if self.shuffle:
                self._rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            self._rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for indices in self._groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def collate_by_n(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack matrices into (batch, n, n) tensors. All items must share n."""
    A = torch.stack([item["A"] for item in batch])
    B = torch.stack([item["B"] for item in batch])
    return {
        "A": A,
        "B": B,
        "n": batch[0]["n"],
        "types": [item["type"] for item in batch],
    }
