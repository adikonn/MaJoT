import torch
import torch.nn as nn


class TriangularizerModel(nn.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n

        # Вход: 2 матрицы n x n -> 2 * n^2 элементов
        in_features = 2 * n * n

        # Выход: количество независимых параметров кососимметричной матрицы
        # Для n=3 это (3 * 2) / 2 = 3 параметра
        out_features = n * (n - 1) // 2

        # Мощный MLP (Многослойный перцептрон) для поиска скрытых зависимостей
        # Используем LayerNorm для стабилизации обучения и GELU для гладкости
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
        ).to(torch.float64)  # Строго переводим веса модели в Double Precision

        # Заранее вычисляем индексы для заполнения матрицы (над диагональю)
        self.row_idx, self.col_idx = torch.triu_indices(n, n, offset=1)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        СТРОГИЙ КОНТРАКТ:
        Вход:
            A: torch.Tensor размерности (Batch, N, N), тип float64
            B: torch.Tensor размерности (Batch, N, N), тип float64
        Выход:
            T: torch.Tensor размерности (Batch, N, N), тип float64
               Ортогональная матрица перехода.
        """
        batch_size = A.shape[0]

        # 1. Вытягиваем матрицы в векторы и склеиваем
        # shape: (Batch, 2 * n^2)
        A_flat = A.reshape(batch_size, -1)
        B_flat = B.reshape(batch_size, -1)
        x = torch.cat([A_flat, B_flat], dim=-1)

        # 2. Нейросеть предсказывает параметры для кососимметричной матрицы
        # shape: (Batch, n*(n-1)/2)
        params = self.net(x)

        # 3. Собираем кососимметричную матрицу K (K^T = -K, на диагонали нули)
        K = torch.zeros(batch_size, self.n, self.n, dtype=A.dtype, device=A.device)

        # Заполняем элементы выше диагонали предсказанными числами
        K[:, self.row_idx, self.col_idx] = params
        # Заполняем элементы ниже диагонали симметрично с минусом
        K[:, self.col_idx, self.row_idx] = -params

        # 4. Вычисляем ортогональную матрицу перехода T через матричную экспоненту.
        # По законам алгебры Ли, экспонента от кососимметричной матрицы
        # ВСЕГДА дает идеальную ортогональную матрицу.
        T = torch.matrix_exp(K)

        return T