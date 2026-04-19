import torch
import torch.nn as nn


class TriangularizerModel(nn.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        in_features = 2 * n * n
        hidden = max(64, 8 * n * n)

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n * n),
        )
        self.to(torch.float64)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Вход:
            A: torch.Tensor размерности (Batch, N, N), тип float64 (Double)
            B: torch.Tensor размерности (Batch, N, N), тип float64 (Double)
        Выход:
            T: torch.Tensor размерности (Batch, N, N), тип float64 (Double)
               Матрица перехода. (Внимание: матрица должна быть обратимой!)
        """
        features = torch.cat([A.reshape(A.shape[0], -1), B.reshape(B.shape[0], -1)], dim=1)
        M = self.encoder(features).reshape(-1, self.n, self.n)

        # exp(M) всегда обратима для любой квадратной M.
        T = torch.linalg.matrix_exp(M)
        return T
