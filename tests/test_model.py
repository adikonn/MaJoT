"""Tests for MatrixTransformer and the training/evaluation utilities.

Run from the project root:
    pytest tests/test_model.py -v

Each test group is tagged with a marker so you can run subsets:
    pytest tests/test_model.py -v -m forward
    pytest tests/test_model.py -v -m losses
    pytest tests/test_model.py -v -m slow
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model
from src.models.matrix_transformer import MatrixTransformer
from src.training.data import (
    GroupedByNBatchSampler,
    MatrixDataset,
    collate_by_n,
)
from src.training.losses import (
    orthogonality_loss,
    total_loss,
    triangularization_loss,
)
from src.evaluation.metrics import evaluate_transform, lower_norm_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(**overrides) -> MatrixTransformer:
    """Return a small but valid MatrixTransformer. Override any kwarg."""
    defaults = dict(hidden_dim=32, num_heads=2, num_layers=2, max_n=16, dropout=0.0)
    defaults.update(overrides)
    return MatrixTransformer(**defaults)


def random_pair(n: int, dtype=torch.float32, device="cpu"):
    """Two random square matrices — no structure assumed."""
    return (
        torch.randn(n, n, dtype=dtype, device=device),
        torch.randn(n, n, dtype=dtype, device=device),
    )


def perfect_pair(n: int, dtype=torch.float32, device="cpu"):
    """A = Q T_A Q^T, B = Q T_B Q^T for random ortho Q and upper-tri T_A, T_B.
    The exact answer is T = Q (L_tri achieves 0)."""
    H = torch.randn(n, n, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(H)
    Q = Q * torch.sign(torch.diag(R))          # Mezzadri: uniform on O(n)
    T_A = torch.triu(torch.randn(n, n, dtype=dtype, device=device))
    T_B = torch.triu(torch.randn(n, n, dtype=dtype, device=device))
    A = Q @ T_A @ Q.T
    B = Q @ T_B @ Q.T
    return A, B, Q


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "forward: forward pass shape/dtype/device tests")
    config.addinivalue_line("markers", "losses: loss function tests")
    config.addinivalue_line("markers", "metrics: evaluation metric tests")
    config.addinivalue_line("markers", "data: dataset / sampler tests")
    config.addinivalue_line("markers", "slow: heavier integration tests (overfit)")


# ===========================================================================
# 1. FORWARD PASS
# ===========================================================================

@pytest.mark.forward
@pytest.mark.parametrize("n", [3, 4, 8])
def test_output_shape_unbatched(n):
    model = make_model()
    A, B = random_pair(n)
    T = model(A, B)
    assert T.shape == (n, n), f"Expected ({n},{n}), got {T.shape}"


@pytest.mark.forward
@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_output_shape_batched(n, batch):
    model = make_model()
    A = torch.randn(batch, n, n)
    B = torch.randn(batch, n, n)
    T = model(A, B)
    assert T.shape == (batch, n, n), f"Expected ({batch},{n},{n}), got {T.shape}"


@pytest.mark.forward
def test_output_dtype_preserved():
    model = make_model()
    A, B = random_pair(5, dtype=torch.float32)
    T = model(A, B)
    assert T.dtype == torch.float32


@pytest.mark.forward
def test_output_device_cpu():
    model = make_model()
    A, B = random_pair(4, device="cpu")
    T = model(A, B)
    assert T.device.type == "cpu", f"Expected cpu, got {T.device}"


@pytest.mark.forward
@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_variable_n_within_max_n(n):
    """Model handles any n ≤ max_n without error."""
    model = make_model(max_n=16)
    A, B = random_pair(n)
    T = model(A, B)
    assert T.shape[-2:] == (n, n)


@pytest.mark.forward
def test_exceeds_max_n_raises():
    model = make_model(max_n=8)
    A, B = random_pair(10)
    with pytest.raises(ValueError, match="exceeds configured max_n"):
        model(A, B)


@pytest.mark.forward
def test_find_transform_interface():
    """find_transform is the Triangularizer interface; expects unbatched tensors."""
    model = make_model()
    A, B = random_pair(5)
    T = model.find_transform(A, B)
    assert T.shape == (5, 5)
    assert not T.requires_grad


@pytest.mark.forward
def test_no_nan_in_output():
    model = make_model()
    A, B = random_pair(6)
    T = model(A, B)
    assert not torch.isnan(T).any(), "NaN in model output"
    assert not torch.isinf(T).any(), "Inf in model output"


# ===========================================================================
# 2. INITIALIZATION BEHAVIOUR
# ===========================================================================

def test_output_is_close_to_identity_at_init():
    """output_proj is zeroed at init, so T = I + delta with delta ≈ 0."""
    model = make_model()
    model.eval()
    n = 6
    A, B = random_pair(n)
    with torch.no_grad():
        T = model(A, B)
    eye = torch.eye(n)
    # delta should be exactly 0 at init (output_proj weight and bias are zero)
    assert torch.allclose(T, eye, atol=1e-6), (
        f"Expected T ≈ I at init, max deviation: {(T - eye).abs().max():.2e}"
    )


@pytest.mark.parametrize("n", [4, 8])
def test_identity_init_loss_equals_original_residual(n):
    """At init T = I, so triangularization_loss equals tril-residual of A and B themselves."""
    model = make_model()
    model.eval()
    A, B = random_pair(n)
    with torch.no_grad():
        T = model(A, B)                         # T ≈ I
        loss_model = triangularization_loss(T, A, B)

    # Compute expected: ||tril(I^T A I)||^2 + ||tril(I^T B I)||^2
    expected = (
        torch.tril(A, diagonal=-1).pow(2).sum()
        + torch.tril(B, diagonal=-1).pow(2).sum()
    )
    assert torch.allclose(loss_model, expected, atol=1e-5), (
        f"Init loss {loss_model:.6f} ≠ expected {expected:.6f}"
    )


# ===========================================================================
# 3. GRADIENT FLOW
# ===========================================================================

def test_gradient_flows_to_parameters():
    """Loss backward must populate .grad for every parameter."""
    model = make_model()
    model.train()
    A, B = random_pair(5)
    T = model(A, B)
    loss, _ = total_loss(T, A, B)
    loss.backward()

    no_grad = [
        name for name, p in model.named_parameters() if p.grad is None
    ]
    assert not no_grad, f"Parameters with no grad: {no_grad}"


def test_loss_is_differentiable_wrt_T():
    """torch.autograd.grad should work with T as the leaf."""
    n = 5
    A, B = random_pair(n)
    T = torch.eye(n, requires_grad=True)
    loss = triangularization_loss(T, A, B)
    (grad,) = torch.autograd.grad(loss, T)
    assert grad.shape == (n, n)
    assert not torch.isnan(grad).any()


# ===========================================================================
# 4. LOSS FUNCTIONS
# ===========================================================================

@pytest.mark.losses
def test_triangularization_loss_zero_for_upper_triangular():
    """If T^T A T is already upper triangular, the loss is 0."""
    n = 5
    T = torch.eye(n)                                        # identity
    A = torch.triu(torch.randn(n, n))                       # upper triangular
    B = torch.triu(torch.randn(n, n))
    loss = triangularization_loss(T, A, B)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.losses
@pytest.mark.parametrize("n", [4, 8])
def test_triangularization_loss_perfect_case(n):
    """For perfect matrices, the correct Q achieves L_tri = 0."""
    A, B, Q = perfect_pair(n, dtype=torch.float64)
    loss = triangularization_loss(Q, A, B)
    assert float(loss) == pytest.approx(0.0, abs=1e-8), (
        f"Expected 0 with Q, got {float(loss):.2e}"
    )


@pytest.mark.losses
@pytest.mark.parametrize("n", [4, 8])
def test_q_better_than_identity_on_perfect_case(n):
    """On perfect matrices Q gives lower L_tri than the identity."""
    A, B, Q = perfect_pair(n, dtype=torch.float64)
    loss_Q = triangularization_loss(Q, A, B)
    I = torch.eye(n, dtype=torch.float64)
    loss_I = triangularization_loss(I, A, B)
    assert float(loss_Q) < float(loss_I), (
        f"Q ({loss_Q:.4f}) should beat I ({loss_I:.4f}) on perfect matrices"
    )


@pytest.mark.losses
def test_orthogonality_loss_is_zero_for_identity():
    T = torch.eye(6)
    assert float(orthogonality_loss(T)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.losses
def test_orthogonality_loss_is_zero_for_orthogonal_Q():
    H = torch.randn(7, 7, dtype=torch.float64)
    Q, _ = torch.linalg.qr(H)
    assert float(orthogonality_loss(Q)) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.losses
def test_orthogonality_loss_positive_for_random():
    T = torch.randn(6, 6)
    assert float(orthogonality_loss(T)) > 0.0


@pytest.mark.losses
def test_total_loss_returns_components():
    n = 5
    A, B = random_pair(n)
    T = torch.eye(n)
    loss, components = total_loss(T, A, B, lambda_orth=2.0)
    assert "loss_tri" in components
    assert "loss_orth" in components
    expected = components["loss_tri"] + 2.0 * components["loss_orth"]
    assert float(loss) == pytest.approx(expected, rel=1e-5)


@pytest.mark.losses
def test_total_loss_lambda_orth_zero_ignores_orth():
    """With lambda_orth=0, the orthogonality term does not affect the loss."""
    n = 5
    A, B = random_pair(n)
    T = torch.randn(n, n)                       # non-orthogonal
    loss, components = total_loss(T, A, B, lambda_orth=0.0)
    assert float(loss) == pytest.approx(components["loss_tri"], rel=1e-5)


@pytest.mark.losses
@pytest.mark.parametrize("n", [3, 4, 8])
def test_total_loss_is_scalar(n):
    A, B = random_pair(n)
    T = torch.eye(n)
    loss, _ = total_loss(T, A, B)
    assert loss.shape == ()


@pytest.mark.losses
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_total_loss_batched(batch):
    n = 5
    A = torch.randn(batch, n, n)
    B = torch.randn(batch, n, n)
    T = torch.eye(n).unsqueeze(0).expand(batch, n, n)
    loss, _ = total_loss(T, A, B)
    assert loss.shape == ()


# ===========================================================================
# 5. EVALUATION METRICS
# ===========================================================================

@pytest.mark.metrics
def test_lower_norm_ratio_zero_for_upper_triangular():
    M = torch.triu(torch.randn(5, 5))
    assert float(lower_norm_ratio(M)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.metrics
def test_lower_norm_ratio_between_zero_and_one():
    M = torch.randn(7, 7)
    ratio = float(lower_norm_ratio(M))
    assert 0.0 <= ratio <= 1.0


@pytest.mark.metrics
def test_lower_norm_ratio_zero_input():
    M = torch.zeros(4, 4)
    assert float(lower_norm_ratio(M)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.metrics
@pytest.mark.parametrize("n", [4, 8])
def test_evaluate_transform_keys(n):
    A, B = random_pair(n)
    T = torch.eye(n)
    result = evaluate_transform(T, A, B)
    expected_keys = {"lower_ratio_A", "lower_ratio_B", "lower_norm_A", "lower_norm_B",
                     "T_cond", "orth_residual"}
    assert expected_keys.issubset(result.keys())


@pytest.mark.metrics
@pytest.mark.parametrize("n", [4, 8])
def test_evaluate_transform_identity_ratios(n):
    """T = I: A' = A, B' = B — ratio equals the ratio of the original matrices."""
    A, B = random_pair(n)
    T = torch.eye(n)
    result = evaluate_transform(T, A, B)
    expected_ratio_A = float(lower_norm_ratio(A))
    assert result["lower_ratio_A"] == pytest.approx(expected_ratio_A, abs=1e-6)


@pytest.mark.metrics
@pytest.mark.parametrize("n", [4, 6])
def test_evaluate_transform_perfect_case_ratios(n):
    """Q should give lower_ratio ≈ 0 on perfect matrices (T_A and T_B are upper triangular)."""
    A, B, Q = perfect_pair(n, dtype=torch.float64)
    result = evaluate_transform(Q, A, B)
    assert result["lower_ratio_A"] == pytest.approx(0.0, abs=1e-6), (
        f"lower_ratio_A={result['lower_ratio_A']:.2e} for perfect pair"
    )
    assert result["lower_ratio_B"] == pytest.approx(0.0, abs=1e-6), (
        f"lower_ratio_B={result['lower_ratio_B']:.2e} for perfect pair"
    )


@pytest.mark.metrics
def test_evaluate_transform_identity_cond():
    n = 4
    T = torch.eye(n, dtype=torch.float64)
    A, B = random_pair(n, dtype=torch.float64)
    result = evaluate_transform(T, A, B)
    assert result["T_cond"] == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# 6. BUILD_MODEL / REGISTRY
# ===========================================================================

def test_build_model_returns_matrix_transformer():
    cfg = dict(name="matrix_transformer", hidden_dim=32, num_heads=2,
                num_layers=2, max_n=16)
    model = build_model(cfg)
    assert isinstance(model, MatrixTransformer)


def test_build_model_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model({"name": "does_not_exist"})


def test_build_model_config_not_mutated():
    """build_model should not modify the caller's config dict."""
    cfg = dict(name="matrix_transformer", hidden_dim=32, num_heads=2,
                num_layers=2, max_n=16)
    original_keys = set(cfg.keys())
    build_model(cfg)
    assert set(cfg.keys()) == original_keys
    assert "name" in cfg, "build_model must not pop 'name' from the caller's dict"


# ===========================================================================
# 7. DATASET & SAMPLER
# ===========================================================================

@pytest.mark.data
def test_grouped_sampler_all_batches_same_n():
    """Every batch produced by the sampler must contain only one value of n."""
    samples = [
        {"n": n, "type": "perfect", "A": torch.zeros(n, n), "B": torch.zeros(n, n)}
        for n in [4, 4, 4, 6, 6, 8, 8, 8, 8]
    ]
    ds = MatrixDataset(samples)
    sampler = GroupedByNBatchSampler(ds, batch_size=2, shuffle=False)
    for batch_indices in sampler:
        ns_in_batch = {ds[i]["n"] for i in batch_indices}
        assert len(ns_in_batch) == 1, (
            f"Batch mixes different n values: {ns_in_batch}"
        )


@pytest.mark.data
def test_grouped_sampler_covers_all_samples():
    """Union of all batches must cover every index exactly once."""
    samples = [
        {"n": 4 if i < 5 else 6, "type": "perfect",
         "A": torch.zeros(4 if i < 5 else 6, 4 if i < 5 else 6),
         "B": torch.zeros(4 if i < 5 else 6, 4 if i < 5 else 6)}
        for i in range(10)
    ]
    ds = MatrixDataset(samples)
    sampler = GroupedByNBatchSampler(ds, batch_size=3, shuffle=False)
    seen = []
    for batch_indices in sampler:
        seen.extend(batch_indices)
    assert sorted(seen) == list(range(len(samples)))


@pytest.mark.data
def test_collate_by_n_stacks_tensors():
    n = 5
    batch = [
        {"n": n, "type": "perfect", "A": torch.ones(n, n), "B": torch.zeros(n, n)}
        for _ in range(4)
    ]
    result = collate_by_n(batch)
    assert result["A"].shape == (4, n, n)
    assert result["B"].shape == (4, n, n)
    assert result["n"] == n
    assert len(result["types"]) == 4


# ===========================================================================
# 8. CUDA
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n", [4, 8])
def test_forward_on_cuda(n):
    model = make_model().cuda()
    A, B = random_pair(n, device="cuda")
    T = model(A, B)
    assert T.device.type == "cuda"
    assert T.shape == (n, n)
    assert not torch.isnan(T).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_find_transform_on_cuda():
    model = make_model().cuda()
    A, B = random_pair(5, device="cuda")
    T = model.find_transform(A, B)
    assert T.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_loss_on_cuda():
    n = 5
    device = "cuda"
    A, B = random_pair(n, device=device)
    T = torch.eye(n, device=device)
    loss, _ = total_loss(T, A, B)
    assert loss.device.type == "cuda"


# ===========================================================================
# 9. INTEGRATION: overfit one sample
# ===========================================================================

@pytest.mark.slow
@pytest.mark.parametrize("matrix_type", ["perfect", "random"])
def test_loss_decreases_on_one_sample(matrix_type: str):
    """50 gradient steps on a single sample must reduce the total loss by at least 50%.
    This is a sanity check that the model is trainable, not that it's good.
    """
    torch.manual_seed(0)
    n = 6
    if matrix_type == "perfect":
        A, B, _ = perfect_pair(n)
    else:
        A, B = random_pair(n)

    model = make_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        T0 = model(A, B)
        loss_init, _ = total_loss(T0, A, B)
    loss_init_val = float(loss_init)

    for _ in range(50):
        optimizer.zero_grad()
        T = model(A, B)
        loss, _ = total_loss(T, A, B)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        T_final = model(A, B)
        loss_final, _ = total_loss(T_final, A, B)
    loss_final_val = float(loss_final)

    assert loss_final_val < loss_init_val * 0.5, (
        f"[{matrix_type}] Loss did not decrease enough: "
        f"{loss_init_val:.4f} -> {loss_final_val:.4f}"
    )


@pytest.mark.slow
def test_loss_decreases_on_batched_input():
    """Same sanity check for batched forward pass."""
    torch.manual_seed(1)
    n, batch = 5, 8
    A = torch.randn(batch, n, n)
    B = torch.randn(batch, n, n)

    model = make_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        T0 = model(A, B)
        loss_init, _ = total_loss(T0, A, B)

    for _ in range(50):
        optimizer.zero_grad()
        T = model(A, B)
        loss, _ = total_loss(T, A, B)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        T_final = model(A, B)
        loss_final, _ = total_loss(T_final, A, B)

    assert float(loss_final) < float(loss_init) * 0.5, (
        f"Batched loss did not decrease: {float(loss_init):.4f} -> {float(loss_final):.4f}"
    )
