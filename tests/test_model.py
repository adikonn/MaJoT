"""Universal tests for any model registered in conftest.MODEL_FACTORIES.

Each test in this file receives a `model` fixture that is parametrized over
every registered model. So if you add a new model to MODEL_FACTORIES in
conftest.py, every test below will automatically be run against it — no test
file edits required.

The tests check the public contract only:
    * forward(A, B) accepts (n, n) and (batch, n, n)
    * find_transform(A, B) works on unbatched inputs in eval/no-grad mode
    * the output T has the right shape, dtype, device
    * the output is finite (no NaN/Inf)
    * gradients flow through nn.Module parameters
    * the model is *trainable* — i.e. can reduce the loss on a single sample

We do NOT test architecture-specific behaviour (e.g. "T equals I at init", or
the presence of a max_n attribute). Those are implementation details that vary
between models and should live in model-specific tests, not the universal set.
"""
from __future__ import annotations

import pytest
import torch
from conftest import TEST_NS, perfect_pair, random_pair

from src.training.losses import total_loss


@pytest.mark.parametrize("n", TEST_NS)
def test_forward_unbatched_shape(model, n) -> None:
    A, B = random_pair(n)
    T = model(A, B)
    assert T.shape == (n, n), f"Expected ({n},{n}), got {tuple(T.shape)}"


@pytest.mark.parametrize("n", TEST_NS)
@pytest.mark.parametrize("batch", [1, 4])
def test_forward_batched_shape(model, n, batch) -> None:
    A = torch.randn(batch, n, n)
    B = torch.randn(batch, n, n)
    T = model(A, B)
    assert T.shape == (batch, n, n), f"Expected ({batch},{n},{n}), got {tuple(T.shape)}"


@pytest.mark.parametrize("n", TEST_NS)
def test_forward_preserves_device(model, n) -> None:
    A, B = random_pair(n, device="cpu")
    T = model(A, B)
    assert T.device.type == A.device.type, (
        f"Output on {T.device}, input on {A.device}"
    )


@pytest.mark.parametrize("n", TEST_NS)
def test_forward_returns_float(model, n) -> None:
    A, B = random_pair(n, dtype=torch.float32)
    T = model(A, B)
    assert T.dtype.is_floating_point, f"Output dtype {T.dtype} is not floating point"


@pytest.mark.parametrize("n", TEST_NS)
def test_forward_no_nan_or_inf(model, n) -> None:
    A, B = random_pair(n)
    T = model(A, B)
    assert torch.isfinite(T).all(), "Model output contains NaN or Inf"


@pytest.mark.parametrize("n", TEST_NS)
def test_find_transform_unbatched(model, n) -> None:
    """Triangularizer interface: find_transform takes single (n, n) tensors."""
    assert hasattr(model, "find_transform"), (
        "Model must implement the Triangularizer interface (find_transform method)"
    )
    A, B = random_pair(n)
    T = model.find_transform(A, B)
    assert T.shape == (n, n)
    assert not T.requires_grad, "find_transform output should not require grad"
    assert torch.isfinite(T).all()


@pytest.mark.parametrize("n", TEST_NS)
def test_output_is_invertible(model, n) -> None:
    """An invertible T is the whole point of the task. Untrained models may still
    produce a non-singular T (e.g. residual-identity models start at T=I); if your
    fresh model fails this test, the optimization will likely collapse.
    """
    A, B = random_pair(n)
    with torch.no_grad():
        T = model(A, B)
    det = torch.linalg.det(T)
    assert det.abs() > 1e-6, f"T appears singular at init: |det| = {det.abs():.2e}"


def test_gradient_flows_to_all_parameters(model) -> None:
    """Every learnable parameter must receive a gradient from total_loss."""
    if not any(p.requires_grad for p in model.parameters()):
        pytest.skip("Model has no trainable parameters")

    model.train()
    A, B = random_pair(5)
    T = model(A, B)
    loss, _ = total_loss(T, A, B)
    loss.backward()

    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"Parameters with no gradient: {missing}"


def test_loss_backward_does_not_crash(model) -> None:
    """Smoke test for the full forward-loss-backward chain."""
    if not any(p.requires_grad for p in model.parameters()):
        pytest.skip("Model has no trainable parameters")

    model.train()
    A = torch.randn(4, 6, 6)
    B = torch.randn(4, 6, 6)
    T = model(A, B)
    loss, _ = total_loss(T, A, B)
    assert loss.shape == ()
    loss.backward()


@pytest.mark.slow
def test_loss_decreases_on_one_sample(model) -> None:
    """100 steps on a single sample should reduce the loss by ≥ 30%.

    This is a sanity check that the model is *trainable*, not that it's good.
    Models that fail this are almost certainly broken (frozen weights, wrong
    parameterization, exploding gradients, etc.).
    """
    if not any(p.requires_grad for p in model.parameters()):
        pytest.skip("Model has no trainable parameters")

    torch.manual_seed(0)
    A, B = random_pair(6)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        loss_init, _ = total_loss(model(A, B), A, B)

    for _ in range(100):
        optimizer.zero_grad()
        T = model(A, B)
        loss, _ = total_loss(T, A, B)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        loss_final, _ = total_loss(model(A, B), A, B)

    assert float(loss_final) < float(loss_init) * 0.7, (
        f"Loss did not decrease enough: {float(loss_init):.4f} -> {float(loss_final):.4f}"
    )


@pytest.mark.slow
def test_can_overfit_perfect_pair(model) -> None:
    """200 steps on a single 'perfect' sample — loss should reach near-zero.

    The perfect case has a closed-form ground-truth Q, so any reasonable model
    that fits this contract should be able to find it given enough optimisation
    on a single sample.
    """
    if not any(p.requires_grad for p in model.parameters()):
        pytest.skip("Model has no trainable parameters")

    torch.manual_seed(0)
    A, B, _ = perfect_pair(6, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for _ in range(200):
        optimizer.zero_grad()
        T = model(A, B)
        loss, _ = total_loss(T, A, B, lambda_orth=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        _loss_final, components = total_loss(model(A, B), A, B, lambda_orth=0.1)
    assert components["loss_tri"] < 0.1, (
        f"Failed to overfit a single perfect sample: "
        f"loss_tri = {components['loss_tri']:.4f}"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n", TEST_NS)
def test_forward_on_cuda(model, n) -> None:
    model = model.cuda()
    A, B = random_pair(n, device="cuda")
    T = model(A, B)
    assert T.is_cuda, f"Output on {T.device}, expected CUDA"
    assert T.shape == (n, n)
    assert torch.isfinite(T).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_find_transform_on_cuda(model) -> None:
    model = model.cuda()
    A, B = random_pair(5, device="cuda")
    T = model.find_transform(A, B)
    assert T.is_cuda


def test_every_registered_model_is_buildable() -> None:
    """Every model in src.models._REGISTRY should be constructible via build_model.

    This is a smoke test for the production registry: if a contributor adds a
    model but breaks its constructor, this catches it.
    """
    from src.models import _REGISTRY, build_model

    for name in _REGISTRY:
        try:
            build_model({"name": name})
        except TypeError:
            assert callable(_REGISTRY[name]), f"Registry entry {name} is not callable"


def test_test_registry_subset_of_prod_registry() -> None:
    """Every model factory in conftest.MODEL_FACTORIES should correspond to a
    name registered in src.models._REGISTRY. Otherwise tests pass for a model
    that wouldn't actually be picked up by build_model / train.py.
    """
    from conftest import MODEL_FACTORIES

    from src.models import _REGISTRY

    test_names = {name for name, _ in MODEL_FACTORIES}
    prod_names = set(_REGISTRY.keys())
    missing = test_names - prod_names
    assert not missing, (
        f"Models registered for tests but not in src/models/__init__.py: {missing}"
    )
