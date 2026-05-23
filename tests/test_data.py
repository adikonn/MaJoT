"""Tests for the data pipeline: MatrixDataset, GroupedByNBatchSampler, collate_by_n."""
from __future__ import annotations

import pytest
import torch

from src.training.data import (
    GroupedByNBatchSampler,
    MatrixDataset,
    collate_by_n,
)


# ---------------------------------------------------------------------------
# MatrixDataset basics
# ---------------------------------------------------------------------------
def test_dataset_len_and_getitem():
    samples = [
        {"n": 4, "type": "perfect", "A": torch.zeros(4, 4), "B": torch.zeros(4, 4)},
        {"n": 6, "type": "noisy",   "A": torch.zeros(6, 6), "B": torch.zeros(6, 6)},
    ]
    ds = MatrixDataset(samples)
    assert len(ds) == 2
    assert ds[0]["n"] == 4
    assert ds[1]["type"] == "noisy"


# ---------------------------------------------------------------------------
# GroupedByNBatchSampler
# ---------------------------------------------------------------------------
def _make_dataset(sizes: list[int]) -> MatrixDataset:
    return MatrixDataset([
        {"n": n, "type": "perfect", "A": torch.zeros(n, n), "B": torch.zeros(n, n)}
        for n in sizes
    ])


def test_sampler_each_batch_has_single_n():
    """The fundamental invariant: no batch mixes matrices of different sizes."""
    ds = _make_dataset([4, 4, 4, 6, 6, 8, 8, 8, 8])
    sampler = GroupedByNBatchSampler(ds, batch_size=2, shuffle=False)
    for batch in sampler:
        ns = {ds[i]["n"] for i in batch}
        assert len(ns) == 1, f"Batch {batch} mixes sizes {ns}"


def test_sampler_covers_all_samples_exactly_once():
    ds = _make_dataset([4] * 5 + [6] * 5)
    sampler = GroupedByNBatchSampler(ds, batch_size=3, shuffle=False)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(len(ds)))


def test_sampler_len_matches_iteration():
    ds = _make_dataset([4] * 7 + [6] * 5)
    sampler = GroupedByNBatchSampler(ds, batch_size=3, shuffle=False)
    actual = sum(1 for _ in sampler)
    assert len(sampler) == actual


def test_sampler_drop_last_drops_partial_batches():
    """With drop_last=True, batches smaller than batch_size are dropped."""
    ds = _make_dataset([4] * 5)                   # 5 samples, batch=2
    sampler = GroupedByNBatchSampler(ds, batch_size=2, shuffle=False, drop_last=True)
    batches = list(sampler)
    assert all(len(b) == 2 for b in batches)
    assert len(batches) == 2                       # 5 // 2 = 2 full batches


def test_sampler_no_drop_keeps_partial_batches():
    ds = _make_dataset([4] * 5)
    sampler = GroupedByNBatchSampler(ds, batch_size=2, shuffle=False, drop_last=False)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 5       # all samples retained


def test_sampler_shuffle_is_deterministic_with_seed():
    """Same seed -> identical batch sequences."""
    ds = _make_dataset([4] * 6 + [6] * 6)
    s1 = list(GroupedByNBatchSampler(ds, batch_size=2, shuffle=True, seed=42))
    s2 = list(GroupedByNBatchSampler(ds, batch_size=2, shuffle=True, seed=42))
    assert s1 == s2


def test_sampler_different_seeds_give_different_orders():
    ds = _make_dataset([4] * 10 + [6] * 10)
    s1 = list(GroupedByNBatchSampler(ds, batch_size=2, shuffle=True, seed=1))
    s2 = list(GroupedByNBatchSampler(ds, batch_size=2, shuffle=True, seed=2))
    assert s1 != s2


def test_sampler_handles_single_size():
    ds = _make_dataset([5] * 8)
    sampler = GroupedByNBatchSampler(ds, batch_size=3, shuffle=False)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 8


# ---------------------------------------------------------------------------
# collate_by_n
# ---------------------------------------------------------------------------
def test_collate_stacks_into_batch_tensor():
    n, batch = 5, 4
    batch_list = [
        {"n": n, "type": "perfect",
         "A": torch.full((n, n), float(i)),
         "B": torch.full((n, n), -float(i))}
        for i in range(batch)
    ]
    out = collate_by_n(batch_list)
    assert out["A"].shape == (batch, n, n)
    assert out["B"].shape == (batch, n, n)
    assert out["n"] == n
    assert out["types"] == ["perfect"] * batch
    # Spot-check that the i-th sample landed in slot i.
    assert out["A"][2, 0, 0].item() == 2.0


def test_collate_preserves_types_list():
    n = 4
    batch_list = [
        {"n": n, "type": t, "A": torch.zeros(n, n), "B": torch.zeros(n, n)}
        for t in ["perfect", "noisy", "random"]
    ]
    out = collate_by_n(batch_list)
    assert out["types"] == ["perfect", "noisy", "random"]


def test_collate_single_item_batch():
    n = 3
    out = collate_by_n([
        {"n": n, "type": "perfect", "A": torch.zeros(n, n), "B": torch.zeros(n, n)}
    ])
    assert out["A"].shape == (1, n, n)


# ---------------------------------------------------------------------------
# Integration with PyTorch DataLoader
# ---------------------------------------------------------------------------
def test_dataloader_end_to_end():
    """Sanity: the sampler + collate combo works through torch.utils.data.DataLoader."""
    from torch.utils.data import DataLoader

    ds = _make_dataset([4] * 6 + [6] * 4)
    loader = DataLoader(
        ds,
        batch_sampler=GroupedByNBatchSampler(ds, batch_size=2, shuffle=False),
        collate_fn=collate_by_n,
    )
    for batch in loader:
        assert batch["A"].dim() == 3                # (batch, n, n)
        assert batch["A"].shape == batch["B"].shape
        n = batch["n"]
        assert batch["A"].shape[1:] == (n, n)
