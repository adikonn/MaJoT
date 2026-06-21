import torch

MATRIX_TYPES = ("perfect", "noisy", "random")
DEFAULT_NOISE_LEVEL = 1e-3
def generate_perfect(n, dtype=torch.float32):
    H = torch.randn(n, n, dtype=dtype)
    Q, R = torch.linalg.qr(H)
    Q = Q * torch.sign(torch.diag(R))
    T_A = torch.triu(torch.randn(n, n, dtype=dtype))
    T_B = torch.triu(torch.randn(n, n, dtype=dtype))
    matrix_a = Q @ T_A @ Q.T
    matrix_b = Q @ T_B @ Q.T
    return matrix_a, matrix_b


def generate_noisy(n, noise_level=1e-3, dtype=torch.float32):
    matrix_a, matrix_b = generate_perfect(n, dtype=dtype)
    matrix_a += noise_level * matrix_a.std() * torch.randn(n, n, dtype=dtype)
    matrix_b += noise_level * matrix_b.std() * torch.randn(n, n, dtype=dtype)
    return matrix_a, matrix_b


def generate_random(n, dtype=torch.float32):
    matrix_a = torch.randn(n, n, dtype=dtype)
    matrix_b = torch.randn(n, n, dtype=dtype)
    return matrix_a, matrix_b

def generate_synthetic_pair(matrix_type, size, noise_level=DEFAULT_NOISE_LEVEL, dtype=torch.float32):
    if matrix_type == "perfect":
        return generate_perfect(size, dtype=dtype)
    if matrix_type == "noisy":
        return generate_noisy(size, noise_level=noise_level, dtype=dtype)
    if matrix_type == "random":
        return generate_random(size, dtype=dtype)
    msg = f"Unknown matrix type: {matrix_type}"
    raise ValueError(msg)
