import torch
import torch.nn as nn


class TriangularizerModel(nn.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        # ВАЖНО: Все веса модели должны быть во float64!
        self.net = nn.Sequential(
            nn.Linear(2 * n * n, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n * n)
        ).to(torch.float64)
        
        # Initialize final layer so T starts close to identity
        nn.init.zeros_(self.net[-1].weight)
        identity_bias = torch.eye(n, dtype=torch.float64).flatten()
        with torch.no_grad():
            self.net[-1].bias.copy_(identity_bias)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Вход:
            A: torch.Tensor размерности (Batch, N, N), тип float64 (Double)
            B: torch.Tensor размерности (Batch, N, N), тип float64 (Double)
        Выход:
            T: torch.Tensor размерности (Batch, N, N), тип float64 (Double)
               Матрица перехода. (Внимание: матрица должна быть обратимой!)
        """
        batch_size = A.shape[0]

        x = torch.cat([A.view(batch_size, -1), B.view(batch_size, -1)], dim=-1)
        out = self.net(x)
        T = out.view(batch_size, self.n, self.n)

        return T
