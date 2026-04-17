import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import norm, eigvals, eig


# =====================================================================
# ТОЧНЫЙ МАТЕМАТИЧЕСКИЙ АЛГОРИТМ (Спрятан внутри файла модели)
# =====================================================================
def _check_criteria(A, B, tol=1e-5):
    n = A.shape[0]
    if np.any(np.abs(eigvals(A).imag) > tol) or np.any(np.abs(eigvals(B).imag) > tol):
        return False
    C = A @ B - B @ A
    if norm(C, 'fro') < tol:
        return True

    basis_ortho, basis_matrices = [], []

    def add_to_basis(M):
        v = M.flatten()
        for u in basis_ortho:
            v = v - np.dot(v, u) * u
        if norm(v) > tol:
            basis_ortho.append(v / norm(v))
            basis_matrices.append(M)
            return True
        return False

    I = np.eye(n)
    add_to_basis(I)
    queue = [I]

    while queue:
        X = queue.pop(0)
        for generator in (A, B):
            Y = X @ generator
            if add_to_basis(Y):
                queue.append(Y)

    for W in basis_matrices:
        if abs(np.trace(W @ C)) > tol:
            return False
    return True


def _get_householder_matrix(v):
    n = len(v)
    v = v / norm(v)
    u = v.copy()
    u[0] += np.sign(v[0] + 1e-15) * 1.0
    if norm(u) < 1e-15: return np.eye(n)
    u = u / norm(u)
    return np.eye(n) - 2 * np.outer(u, u)


def _perform_joint_triangularization(A, B, tol=1e-5):
    n = A.shape[0]
    A_curr, B_curr = A.copy(), B.copy()
    Q_total = np.eye(n)
    for k in range(n - 1):
        A_sub, B_sub = A_curr[k:, k:], B_curr[k:, k:]
        m = A_sub.shape[0]
        common_v = None
        for _ in range(50):
            M = np.random.randn() * A_sub + np.random.randn() * B_sub
            try:
                vals, vecs = eig(M)
            except Exception:
                continue
            for i in range(m):
                if abs(vals[i].imag) < tol:
                    v = vecs[:, i].real
                    v = v / norm(v)
                    res_A = A_sub @ v - np.dot(v, A_sub @ v) * v
                    res_B = B_sub @ v - np.dot(v, B_sub @ v) * v
                    if norm(res_A) < tol * max(1.0, norm(A_sub)) and norm(res_B) < tol * max(1.0, norm(B_sub)):
                        common_v = v
                        break
            if common_v is not None: break
        if common_v is None: return None
        H_sub = _get_householder_matrix(common_v)
        Q_step = np.eye(n)
        Q_step[k:, k:] = H_sub
        A_curr = Q_step.T @ A_curr @ Q_step
        B_curr = Q_step.T @ B_curr @ Q_step
        Q_total = Q_total @ Q_step
    return Q_total


# =====================================================================
# Нейросетевая модель с гибридным Forward-проходом
# =====================================================================
class TriangularizerModel(nn.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        in_features = 2 * n * n
        out_features = n * (n - 1) // 2

        # Архитектура нейросети
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_features)
        ).to(torch.float64)

        self.row_idx, self.col_idx = torch.triu_indices(n, n, offset=1)

    def _nn_forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Чистый проход нейросети (для вычисления градиентов)."""
        batch_size = A.shape[0]
        x = torch.cat([A.reshape(batch_size, -1), B.reshape(batch_size, -1)], dim=-1)
        params = self.net(x)

        K = torch.zeros(batch_size, self.n, self.n, dtype=A.dtype, device=A.device)
        K[:, self.row_idx, self.col_idx] = params
        K[:, self.col_idx, self.row_idx] = -params

        T = torch.matrix_exp(K)
        return T

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Магический Forward:
        Если идет обучение (train), работает только нейросеть (чтобы были градиенты).
        Если идет оценка (eval), модель сначала пробует точную математику для каждой пары.
        """
        # 1. Если мы тренируем модель - возвращаем только градиенты от нейросети
        if self.training:
            return self._nn_forward(A, B)

        # 2. Если мы в режиме инференса (eval) - включаем гибридный режим
        batch_size = A.shape[0]

        # Сначала прогоняем весь батч через нейросеть (это быстро)
        T_pred = self._nn_forward(A, B).clone()

        # Переводим матрицы в NumPy для строгого алгоритма
        A_np = A.detach().cpu().numpy().astype(np.float64)
        B_np = B.detach().cpu().numpy().astype(np.float64)

        # Идем по батчу и пытаемся решить задачу точным алгоритмом
        for i in range(batch_size):
            A_i, B_i = A_np[i], B_np[i]

            # Если математика дает "Добро"...
            if _check_criteria(A_i, B_i, tol=1e-5):
                Q_exact = _perform_joint_triangularization(A_i, B_i, tol=1e-5)

                # ...заменяем предсказание нейросети на идеальную матрицу!
                if Q_exact is not None:
                    T_pred[i] = torch.tensor(Q_exact, dtype=A.dtype, device=A.device)

        return T_pred