"""Tests for evaluation metrics. Model-agnostic."""
from __future__ import annotations

import pytest
import torch

from src.evaluation.metrics import evaluate_transform, lower_norm_ratio

from conftest import perfect_pair, random_pair


# ---------------------------------------------------------------------------
# lower_norm_ratio
# ---------------------------------------------------------------------------
def test_ratio_zero_for_upper_triangular():
    M = torch.triu(torch.randn(5, 5))
    assert float(lower_norm_ratio(M)) == pytest.approx(0.0, abs=1e-6)


def test_ratio_one_for_strictly_lower_triangular():
    M = torch.tril(torch.randn(5, 5), diagonal=-1)
    assert float(lower_norm_ratio(M)) == pytest.approx(1.0, abs=1e-6)


def test_ratio_in_unit_interval():
    for _ in range(10):
        ratio = float(lower_norm_ratio(torch.randn(7, 7)))
        assert 0.0 <= ratio <= 1.0


def test_ratio_zero_for_zero_matrix():
    """Edge case: avoid division by zero, return 0."""
    assert float(lower_norm_ratio(torch.zeros(4, 4))) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# evaluate_transform
# ---------------------------------------------------------------------------
EXPECTED_KEYS = {
    "lower_ratio_A",
    "lower_ratio_B",
    "lower_norm_A",
    "lower_norm_B",
    "T_cond",
    "orth_residual",
}


@pytest.mark.parametrize("n", [4, 8])
def test_evaluate_returns_expected_keys(n):
    A, B = random_pair(n)
    result = evaluate_transform(torch.eye(n), A, B)
    assert EXPECTED_KEYS.issubset(result.keys())


def test_evaluate_returns_python_floats():
    """Logging to wandb / CSV requires Python floats, not tensors."""
    n = 5
    A, B = random_pair(n)
    result = evaluate_transform(torch.eye(n), A, B)
    for k, v in result.items():
        assert isinstance(v, float), f"{k} is {type(v)}, not float"


@pytest.mark.parametrize("n", [4, 8])
def test_evaluate_identity_matches_input_ratios(n):
    """T = I means A' = A, B' = B, so the ratios equal those of the inputs."""
    A, B = random_pair(n)
    result = evaluate_transform(torch.eye(n), A, B)
    assert result["lower_ratio_A"] == pytest.approx(float(lower_norm_ratio(A)), abs=1e-6)
    assert result["lower_ratio_B"] == pytest.approx(float(lower_norm_ratio(B)), abs=1e-6)


@pytest.mark.parametrize("n", [4, 8])
def test_evaluate_perfect_case_with_ground_truth_Q(n):
    """Q applied to a perfect pair gives lower_ratio ≈ 0."""
    A, B, Q = perfect_pair(n, dtype=torch.float64)
    result = evaluate_transform(Q, A, B)
    assert result["lower_ratio_A"] == pytest.approx(0.0, abs=1e-6)
    assert result["lower_ratio_B"] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_identity_cond_is_one():
    n = 4
    A, B = random_pair(n, dtype=torch.float64)
    result = evaluate_transform(torch.eye(n, dtype=torch.float64), A, B)
    assert result["T_cond"] == pytest.approx(1.0, abs=1e-4)


def test_evaluate_identity_orth_residual_is_zero():
    n = 4
    A, B = random_pair(n)
    result = evaluate_transform(torch.eye(n), A, B)
    assert result["orth_residual"] == pytest.approx(0.0, abs=1e-5)


def test_evaluate_handles_singular_T_gracefully():
    """A singular T should not crash evaluation — cond might be inf, but the
    rest of the metrics should still be valid Python floats.
    """
    n = 4
    A, B = random_pair(n)
    # A rank-deficient T (last column = first column).
    T = torch.randn(n, n)
    T[:, -1] = T[:, 0]
    result = evaluate_transform(T, A, B)
    for k, v in result.items():
        assert isinstance(v, float), f"{k} = {v!r} is not a float"
