# optim_newton.py

import torch


def joint_triangularize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Бейзлайн 2: оптимизация на ортогональной группе.

    Args:
        A: torch.Tensor, квадратная матрица размера (n, n)
        B: torch.Tensor, квадратная матрица размера (n, n)

    Returns:
        Q: torch.Tensor, ортогональная матрица размера (n, n)
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]
    dtype, device = A.dtype, A.device

    # Заглушка
    X = torch.randn(n, n, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(X)

    return Q
