"""Shared test fixtures and the model-under-test registry.

================================================================================
HOW TO ADD YOUR MODEL TO THE TEST SUITE
================================================================================
1. Implement your model in src/models/ as an nn.Module that also implements the
   Triangularizer interface (i.e. provides `find_transform(A, B) -> T` and a
   normal `forward(A, B) -> T`).
2. Register your model in `src/models/__init__.py` so `build_model` knows it.
3. Add an entry to MODEL_FACTORIES below — a tuple of (display_name, factory).
   The factory must construct a SMALL instance of your model (training-test
   instances are intentionally tiny so the full suite stays fast).

After step 3 every test in test_model.py will be automatically parametrized
over your new model. No further edits to test files are required.
================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cross_attn_triangularizer import CrossAttnTriangularizer
from src.models.dual_stream_rowcol import DualStreamRowCol
from src.models.dual_stream_rowcol_ortho import DualStreamRowColOrtho
from src.models.equivariant_matrix_net import EquivariantMatrixNet
from src.models.iterative_refinement import IterativeRefinementTriangularizer
from src.models.iterative_refinement_ortho import IterativeRefinementOrtho
from src.models.learned_givens import LearnedGivens
from src.models.matrix_transformer_ortho import MatrixTransformerOrtho

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def _factory_dual_stream_rowcol() -> nn.Module:
    return DualStreamRowCol(hidden_dim=32, num_heads=2, num_layers=2, max_n=16)


def _factory_dual_stream_rowcol_ortho() -> nn.Module:
    return DualStreamRowColOrtho(hidden_dim=32, num_heads=2, num_layers=2, max_n=16)


def _factory_iterative_refinement() -> nn.Module:
    return IterativeRefinementTriangularizer(hidden_dim=32, num_heads=2, num_steps=4, max_n=16)


def _factory_iterative_refinement_ortho() -> nn.Module:
    return IterativeRefinementOrtho(hidden_dim=32, num_heads=2, num_steps=4, max_n=16)


def _factory_matrix_transformer_ortho() -> nn.Module:
    return MatrixTransformerOrtho(hidden_dim=32, num_heads=2, num_layers=2, max_n=16)


def _factory_equivariant_matrix_net() -> nn.Module:
    return EquivariantMatrixNet(hidden_dim=16, num_layers=2)


def _factory_cross_attn_triangularizer() -> nn.Module:
    return CrossAttnTriangularizer(hidden_dim=16, num_heads=2)
def _factory_learned_givens() -> nn.Module:
    # Keep it small for tests.
    return LearnedGivens(hidden_dim=32, num_heads=2, num_layers=2, max_n=16, num_rotations=32)


MODEL_FACTORIES: list[tuple[str, Callable[[], nn.Module]]] = [
    ("dual_stream_rowcol", _factory_dual_stream_rowcol),
    ("dual_stream_rowcol_ortho", _factory_dual_stream_rowcol_ortho),
    ("iterative_refinement", _factory_iterative_refinement),
    ("iterative_refinement_ortho", _factory_iterative_refinement_ortho),
    ("matrix_transformer_ortho", _factory_matrix_transformer_ortho),
    ("equivariant_matrix_net", _factory_equivariant_matrix_net),
    ("cross_attn_triangularizer", _factory_cross_attn_triangularizer),
    ("learned_givens", _factory_learned_givens),
]

# Matrix sizes used by parametrized universal tests. Any registered model must
# at minimum support these. Adjust if you intentionally want larger coverage.
TEST_NS: list[int] = [4, 8]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(
    params=[factory for _, factory in MODEL_FACTORIES],
    ids=[name for name, _ in MODEL_FACTORIES],
)
def model(request) -> nn.Module:
    """A freshly constructed instance of each registered model.

    Use this in any test that should be run against every model.
    """
    return request.param()


@pytest.fixture
def model_name(request) -> str:
    """Display name of the current model being tested (matches the fixture id)."""
    return request.node.callspec.id if hasattr(request.node, "callspec") else "unknown"


# ---------------------------------------------------------------------------
# Shared helpers used across multiple test files
# ---------------------------------------------------------------------------
def random_pair(n: int, dtype=torch.float32, device: str = "cpu"):
    """Two unrelated random matrices."""
    return (
        torch.randn(n, n, dtype=dtype, device=device),
        torch.randn(n, n, dtype=dtype, device=device),
    )


def perfect_pair(n: int, dtype=torch.float64, device: str = "cpu"):
    """A pair (A, B, Q) such that Q.T @ A @ Q and Q.T @ B @ Q are both upper triangular.

    Returns the ground-truth Q together with the matrices.
    """
    H = torch.randn(n, n, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(H)
    Q = Q * torch.sign(torch.diag(R))            # Mezzadri: uniform on O(n)
    T_A = torch.triu(torch.randn(n, n, dtype=dtype, device=device))
    T_B = torch.triu(torch.randn(n, n, dtype=dtype, device=device))
    A = Q @ T_A @ Q.T
    B = Q @ T_B @ Q.T
    return A, B, Q


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: heavier integration tests (overfit, training)")
    config.addinivalue_line("markers", "cuda: requires CUDA device")