from __future__ import annotations
import torch
import scipy.linalg

def joint_triangularize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape == B.shape
    
    n = A.shape[0]
    device = A.device
    work_dtype = torch.float64
    
    A_64 = A.to(dtype=work_dtype, device=device)
    B_64 = B.to(dtype=work_dtype, device=device)
    
    eps = 1e-9
    A_reg = A_64 + torch.eye(n, device=device, dtype=work_dtype) * eps
    B_reg = B_64 + torch.eye(n, device=device, dtype=work_dtype) * eps
    
    A_np = A_reg.detach().cpu().numpy()
    B_np = B_reg.detach().cpu().numpy()
    
    try:
        _, eigvecs = scipy.linalg.eig(A_np, B_np)
        T_res = torch.from_numpy(eigvecs).to(device=device, dtype=work_dtype)
    except Exception:
        M_np = 0.5 * A_np + 0.5 * B_np
        _, Z_mat = scipy.linalg.schur(M_np, output='complex')
        T_res = torch.from_numpy(Z_mat).to(device=device, dtype=work_dtype)

    return T_res.real
