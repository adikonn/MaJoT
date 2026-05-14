import pytest
import torch
import numpy as np

# Импортируем все бейзлайны
from baseline.pencil_schur import joint_triangularize as jt_matrix_pencil
from baseline.jacobi_type import joint_triangularize as jt_jacobi
from baseline.optim_newton import joint_triangularize as jt_newton

BASELINES = [
    ("schur", jt_schur),
    ("jacobi", jt_jacobi),
    ("newton", jt_newton)
]

def generate_test_matrices(n, device="cpu", dtype=torch.float64):
    # Генерируем случайные квадратные матрицы
    A = torch.randn(n, n, device=device, dtype=dtype)
    B = torch.randn(n, n, device=device, dtype=dtype)
    return A, B

def get_lower_triangular_residual(M):
    """Вычисляет сумму квадратов элементов строго ниже главной диагонали."""
    tril_indices = torch.tril_indices(M.shape[0], M.shape[1], offset=-1)
    return torch.sum(M[tril_indices[0], tril_indices[1]] ** 2)

@pytest.mark.parametrize("name, joint_triangularize", BASELINES)
@pytest.mark.parametrize("n", [4, 10])
def test_returns_orthogonal_matrix(name, joint_triangularize, n):
    A, B = generate_test_matrices(n)
    
    Q = joint_triangularize(A, B)
    
    assert Q.shape == (n, n), f"{name}: Q has wrong shape"
    assert Q.device == A.device, f"{name}: Q is on wrong device"
    assert Q.dtype == A.dtype, f"{name}: Q has wrong dtype"
    
    # Проверка на ортогональность: Q^T @ Q ≈ I
    I = torch.eye(n, dtype=Q.dtype, device=Q.device)
    assert torch.allclose(Q.T @ Q, I, atol=1e-5), f"{name}: Q is not orthogonal"


@pytest.mark.parametrize("name, joint_triangularize", BASELINES)
def test_minimizes_residual_vs_random(name, joint_triangularize):
    """
    Проверяет, что результаты алгоритма (A', B') имеют меньший или 
    соизмеримый residual нижнетреугольных элементов по сравнению 
    со случайным ортогональным преобразованием.
    """
    n = 8
    A, B = generate_test_matrices(n)
    
    # Решение алгоритма
    Q = joint_triangularize(A, B)
    A_prime = Q.T @ A @ Q
    B_prime = Q.T @ B @ Q
    
    res_alg = get_lower_triangular_residual(A_prime) + get_lower_triangular_residual(B_prime)
    
    # Случайное ортогональное преобразование
    H = torch.randn(n, n, dtype=A.dtype, device=A.device)
    Q_rand, _ = torch.linalg.qr(H)
    A_rand = Q_rand.T @ A @ Q_rand
    B_rand = Q_rand.T @ B @ Q_rand
    
    res_rand = get_lower_triangular_residual(A_rand) + get_lower_triangular_residual(B_rand)
    
    # Бейзлайн должен хоть как-то стараться минимизировать нижнетреугольную часть.
    # Это условие может нарушаться для единичных случайных матриц, если бейзлайн неэффективен, 
    # но оно работает как базовый sanity check.
    assert res_alg <= res_rand * 1.5, f"{name}: Residual is too large {res_alg} vs random {res_rand}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("name, joint_triangularize", BASELINES)
def test_cuda_support(name, joint_triangularize):
    n = 6
    A, B = generate_test_matrices(n, device="cuda")
    Q = joint_triangularize(A, B)
    
    assert Q.is_cuda, f"{name}: Q should be on CUDA"
    I = torch.eye(n, dtype=Q.dtype, device=Q.device)
    assert torch.allclose(Q.T @ Q, I, atol=1e-5), f"{name}: Q is not orthogonal on CUDA"
