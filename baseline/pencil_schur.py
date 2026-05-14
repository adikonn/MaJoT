import torch
import scipy.linalg
import numpy as np

def joint_triangularize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # 1. Проверка входа
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D tensors"
    assert A.shape == B.shape, "A and B must have the same shape"
    assert A.shape[0] == A.shape[1], "A and B must be square matrices"
    
    device = A.device
    dtype = A.dtype
    
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    
    # 2. Построение матрицы-карандаша C = alpha * A + beta * B
    alpha = 1.0
    beta = 0.5
    C_np = alpha * A_np + beta * B_np
    
    # 3. Schur-разложение (вещественное, чтобы получить ортогональную Q)
    # scipy.linalg.schur возвращает (T, Z), где Z - унитарная/ортогональная матрица преобразования (Q)
    T, Q = scipy.linalg.schur(C_np, output='real')
    
    # 5. Возврат результата (Q, с тем же device и dtype)
    return torch.from_numpy(Q).to(device=device, dtype=dtype)
