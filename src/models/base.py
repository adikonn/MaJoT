"""Abstract interface that any solver of the joint triangularization problem must implement.

Both classical baselines and neural network models should expose a `find_transform`
method so that `scripts/benchmark.py` can evaluate them uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Triangularizer(ABC):
    """Find an invertible T such that T^T A T and T^T B T are (close to) upper triangular."""

    name: str = "abstract"

    @abstractmethod
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Return T (n x n) given a single pair of matrices A, B (n x n)."""
        raise NotImplementedError
