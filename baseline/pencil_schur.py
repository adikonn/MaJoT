# pencil_schur.py

import torch


def joint_triangularize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Бейзлайн 1: построить матрицу-карандаш C = A + 0.5*B,
    выполнить Schur-разложение и использовать ортогональный Q как общее преобразование.

    Args:
        A: torch.Tensor, квадратная матрица размера (n, n)
        B: torch.Tensor, квадратная матрица размера (n, n)

    Returns:
        Q: torch.Tensor, ортогональная матрица размера (n, n)
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]

    # Линейная комбинация: C = A + 0.5 * B
    C = A + 0.5 * B

    # Пока просто "заглушка";
    Q = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

    return Q
