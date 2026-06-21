import torch
from torch import nn


class EquivariantLayer(nn.Module):
    """Слой, извлекающий признаки с учетом инвариантности к перестановкам.
    Объединяет локальные признаки элементов (i, j) с глобальными признаками i-той строки и j-того столбца.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim * 3, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_pool = x.mean(dim=3, keepdim=True).expand_as(x)
        col_pool = x.mean(dim=2, keepdim=True).expand_as(x)

        cat = torch.cat([x, row_pool, col_pool], dim=1)
        return self.mlp(cat)


class EquivariantMatrixNet(nn.Module):
    r"""Нейросетевая архитектура для совместной триангуляризации матриц.

    Обоснование:
    1. Идея: Использование 1x1 сверток (эквивалент PointNet/DeepSets) с агрегацией по строкам и столбцам.
    2. Почему подходит: Гарантирует независимость числа параметров от n, что позволяет батчить и инференсить матрицы любого размера. Естественно обрабатывает row/column-структуру.
    3. Предположения: Модель аппроксимирует многообразие ортогональных матриц в $\mathbb{R}^{n \times n}$. В отличие от риманового градиентного спуска на ортогональной группе, сеть предсказывает сырую матрицу за $O(1)$ шагов. Строгая ортогонализация достигается через QR-разложение на инференсе.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.init_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.layers = nn.ModuleList([
            EquivariantLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.final_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            nn.init.zeros_(self.final_proj.bias)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        is_batched = A.dim() == 3
        if not is_batched:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)

        x = torch.stack([A, B], dim=1)
        x = self.init_proj(x)

        for layer in self.layers:
            x = layer(x)

        eye = torch.eye(A.size(-1), device=A.device)
        T = self.final_proj(x).squeeze(1) + eye

        if not is_batched:
            T = T.squeeze(0)

        return T

    @torch.no_grad()
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Инференс-метод (Triangularizer contract).
        Применяет QR-разложение для обеспечения жесткого выполнения T^T T = I.
        """
        T_raw = self.forward(A, B)

        Q, R = torch.linalg.qr(T_raw)
        sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        return Q * sign.unsqueeze(-2) if T_raw.dim() == 3 else Q * sign.unsqueeze(0)

