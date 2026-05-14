import torch

def joint_triangularize(
    A: torch.Tensor, 
    B: torch.Tensor, 
    lr: float = 0.1, 
    max_iter: int = 1000, 
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Бейзлайн 2: оптимизация на ортогональной группе (Riemannian Gradient Descent).

    Ищет ортогональную матрицу Q, минимизирующую сумму квадратов элементов
    (строго нижних или строго верхних треугольных частей) матриц Q^T A Q и Q^T B Q.
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]
    dtype, device = A.dtype, A.device

    X = torch.randn(n, n, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(X)
    Q.requires_grad_(True)

    M = torch.stack([A, B])
    prev_loss = float('inf')

    for _ in range(max_iter):
        Q_ext = Q.unsqueeze(0)
        M_transformed = Q_ext.mT @ M @ Q_ext
        
        # Если по ТЗ матрицы должны в итоге стать НИЖНЕТРЕУГОЛЬНЫМИ, 
        # штрафовать нужно ВЕРХНЮЮ часть: torch.triu(M_transformed, diagonal=1).
        # Если цель - ВЕРХНЕТРЕУГОЛЬНЫЙ вид (как в формуле из ТЗ), оставляем tril:
        off_diag_parts = torch.tril(M_transformed, diagonal=-1) 
        
        # ОПТИМИЗАЦИЯ: убрали дорогой torch.linalg.matrix_norm().pow(2)
        # Квадрат нормы Фробениуса — это просто сумма квадратов всех элементов.
        loss = (off_diag_parts ** 2).sum()

        if torch.isnan(loss) or torch.isinf(loss):
            break

        current_loss = loss.item()
        
        if prev_loss != float('inf'):
            if abs(prev_loss - current_loss) / (prev_loss + 1e-9) < tol:
                break
        prev_loss = current_loss

        loss.backward()
        Z = Q.grad

        # Использование mT вместо T для единообразия и безопасной работы с батчами
        S = 0.5 * (Q.mT @ Z - Z.mT @ Q)

        if torch.linalg.matrix_norm(S) < tol:
            break

        with torch.no_grad():
            step_matrix = torch.linalg.matrix_exp(-lr * S)
            Q_next = Q @ step_matrix

        Q = Q_next.detach().requires_grad_(True)

    with torch.no_grad():
        Q_final, R = torch.linalg.qr(Q.detach())
        
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        
        Q_final = Q_final * signs

    return Q_final