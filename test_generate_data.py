import torch

from dataset.generate_data import MATRIX_TYPES, generate_synthetic_pair


def test_generate_synthetic_pair_supports_size_for_all_types():
    size = 7
    for matrix_type in MATRIX_TYPES:
        a, b = generate_synthetic_pair(matrix_type, size=size)
        assert a.shape == (size, size)
        assert b.shape == (size, size)


def test_generate_synthetic_pair_supports_noise_parameter():
    torch.manual_seed(123)
    perfect_a, perfect_b = generate_synthetic_pair("perfect", size=5)

    torch.manual_seed(123)
    noisy_a, noisy_b = generate_synthetic_pair("noisy", size=5, noise_level=0.0)

    assert torch.allclose(perfect_a, noisy_a)
    assert torch.allclose(perfect_b, noisy_b)

