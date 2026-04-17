import torch
import torch.nn as nn


class TriangularizerModel(nn.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        # ВАЖНО: Все веса модели должны быть во float64!
        # Заглушка
        self.dummy_layer = nn.Linear(1, 1).to(torch.float64)

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

        # Заглушка
        T = torch.eye(self.n, dtype=torch.float64, device=A.device)
        T = T.unsqueeze(0).expand(batch_size, -1, -1)

        return T
