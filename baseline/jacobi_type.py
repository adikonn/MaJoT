# jacobi_type.py

import torch


def joint_triangularize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Бейзлайн 3: Jacobi-подобный итеративный метод.
    Args:
        A: torch.Tensor, квадратная матрица размера (n, n)
        B: torch.Tensor, квадратная матрица размера (n, n)

    Returns:
        Q: torch.Tensor, ортогональная матрица размера (n, n)
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]

    # Заглушка: возвращаем единичную матрицу вместо Q,
    # которую метод должен постепенно строить из вращений.
    Q = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

    return Q
