import scipy.linalg
import torch


def joint_triangularize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D tensors"
    assert A.shape == B.shape, "A and B must have the same shape"
    assert A.shape[0] == A.shape[1], "A and B must be square matrices"

    device = A.device
    dtype = A.dtype

    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()

    alpha = 1.0
    beta = 0.5
    C_np = alpha * A_np + beta * B_np

    schur_result = scipy.linalg.schur(C_np, output="real")
    Q = schur_result[1]

    return torch.from_numpy(Q).to(device=device, dtype=dtype)
