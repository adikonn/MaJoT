"""Tests for the training loss functions.

These tests are model-agnostic: they verify the mathematical correctness of
triangularization_loss, orthogonality_loss, and total_loss. Adding new models
does not change these tests.
"""
from __future__ import annotations

import pytest
import torch
from conftest import perfect_pair, random_pair

from src.training.losses import (
    orthogonality_loss,
    total_loss,
    triangularization_loss,
)


def test_tri_loss_zero_when_T_is_I_and_AB_upper_triangular() -> None:
    n = 5
    T = torch.eye(n)
    A = torch.triu(torch.randn(n, n))
    B = torch.triu(torch.randn(n, n))
    assert float(triangularization_loss(T, A, B)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("n", [4, 8])
def test_tri_loss_zero_for_ground_truth_Q_on_perfect_pair(n) -> None:
    A, B, Q = perfect_pair(n, dtype=torch.float64)
    assert float(triangularization_loss(Q, A, B)) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("n", [4, 8])
def test_ground_truth_Q_beats_identity_on_perfect_pair(n) -> None:
    A, B, Q = perfect_pair(n, dtype=torch.float64)
    I = torch.eye(n, dtype=torch.float64)
    assert float(triangularization_loss(Q, A, B)) < float(triangularization_loss(I, A, B))


@pytest.mark.parametrize("n", [3, 5])
def test_tri_loss_is_scalar(n) -> None:
    A, B = random_pair(n)
    loss = triangularization_loss(torch.eye(n), A, B)
    assert loss.shape == ()


@pytest.mark.parametrize("batch", [1, 4])
def test_tri_loss_batched(batch) -> None:
    n = 5
    A = torch.randn(batch, n, n)
    B = torch.randn(batch, n, n)
    T = torch.eye(n).unsqueeze(0).expand(batch, n, n)
    loss = triangularization_loss(T, A, B)
    assert loss.shape == ()


def test_tri_loss_non_negative() -> None:
    """The loss is a sum of squares — must be ≥ 0 always."""
    for _ in range(10):
        n = torch.randint(2, 10, (1,)).item()
        T = torch.randn(n, n)
        A, B = random_pair(n)
        assert float(triangularization_loss(T, A, B)) >= 0.0


def test_tri_loss_is_differentiable_wrt_T() -> None:
    n = 5
    A, B = random_pair(n)
    T = torch.eye(n, requires_grad=True)
    loss = triangularization_loss(T, A, B)
    (grad,) = torch.autograd.grad(loss, T)
    assert grad.shape == (n, n)
    assert torch.isfinite(grad).all()


def test_orth_loss_zero_for_identity() -> None:
    assert float(orthogonality_loss(torch.eye(6))) == pytest.approx(0.0, abs=1e-6)


def test_orth_loss_zero_for_orthogonal_matrix() -> None:
    H = torch.randn(7, 7, dtype=torch.float64)
    Q, _ = torch.linalg.qr(H)
    assert float(orthogonality_loss(Q)) == pytest.approx(0.0, abs=1e-10)


def test_orth_loss_positive_for_random() -> None:
    assert float(orthogonality_loss(torch.randn(6, 6))) > 0.0


def test_orth_loss_zero_for_negation_of_orthogonal() -> None:
    """-Q is also orthogonal."""
    H = torch.randn(5, 5, dtype=torch.float64)
    Q, _ = torch.linalg.qr(H)
    assert float(orthogonality_loss(-Q)) == pytest.approx(0.0, abs=1e-10)


def test_total_loss_returns_dict_components() -> None:
    n = 5
    A, B = random_pair(n)
    T = torch.eye(n)
    _, components = total_loss(T, A, B, lambda_orth=2.0)
    assert set(components.keys()) == {"loss_tri", "loss_orth"}


def test_total_loss_components_match_total() -> None:
    n = 5
    A, B = random_pair(n)
    T = torch.randn(n, n)
    loss, c = total_loss(T, A, B, lambda_orth=1.5)
    expected = c["loss_tri"] + 1.5 * c["loss_orth"]
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_lambda_orth_zero_ignores_orth_term() -> None:
    n = 5
    A, B = random_pair(n)
    T = torch.randn(n, n)
    loss, c = total_loss(T, A, B, lambda_orth=0.0)
    assert float(loss) == pytest.approx(c["loss_tri"], rel=1e-5)


def test_lambda_orth_scales_orth_term() -> None:
    n = 5
    A, B = random_pair(n)
    T = torch.randn(n, n)
    loss_1, _ = total_loss(T, A, B, lambda_orth=1.0)
    loss_2, _ = total_loss(T, A, B, lambda_orth=2.0)
    _, c = total_loss(T, A, B, lambda_orth=1.0)
    assert float(loss_2 - loss_1) == pytest.approx(c["loss_orth"], rel=1e-4)
