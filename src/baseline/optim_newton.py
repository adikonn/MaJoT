import torch


def joint_triangularize(
    A: torch.Tensor,
    B: torch.Tensor,
    lr: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Бейзлайн 2: оптимизация на ортогональной группе (Riemannian Gradient Descent).

    Ищет ортогональную матрицу Q, минимизирующую сумму квадратов элементов
    (строго нижних или строго верхних треугольных частей) матриц Q^T A Q и Q^T B Q.
    """
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]
    dtype, device = A.dtype, A.device

    x = torch.randn(n, n, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(x)
    q.requires_grad_(True)

    M = torch.stack([A, B])
    prev_loss = float("inf")

    for _ in range(max_iter):
        q_ext = q.unsqueeze(0)
        M_transformed = q_ext.mT @ M @ q_ext

        off_diag_parts = torch.tril(M_transformed, diagonal=-1)

        loss = (off_diag_parts**2).sum()

        if torch.isnan(loss) or torch.isinf(loss):
            break

        current_loss = loss.item()

        if prev_loss != float("inf"):
            if abs(prev_loss - current_loss) / (prev_loss + 1e-9) < tol:
                break
        prev_loss = current_loss

        loss.backward()
        grad = q.grad
        if grad is None:
            break

        skew = 0.5 * (q.mT @ grad - grad.mT @ q)

        if torch.linalg.matrix_norm(skew) < tol:
            break

        with torch.no_grad():
            step_matrix = torch.linalg.matrix_exp(-lr * skew)
            q_next = q @ step_matrix

        q = q_next.detach().requires_grad_(True)

    with torch.no_grad():
        Q_final, R = torch.linalg.qr(q.detach())

        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0

        return Q_final * signs

