import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangularizerModel(nn.Module):
    def __init__(self, n: int = 3, hidden_dim: int = 384, depth: int = 6, dropout: float = 0.10):
        super().__init__()
        self.n = n
        input_dim = 2 * n * n

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(depth)
            ]
        )

        # Параметры для конструирования T = L @ U (обратима из-за exp(diag(U))).
        n_off = n * (n - 1) // 2
        self.l_head = nn.Linear(hidden_dim, n_off)
        self.u_off_head = nn.Linear(hidden_dim, n_off)
        self.u_diag_head = nn.Linear(hidden_dim, n)

        self.to(torch.float64)

        # Инициализация около identity, чтобы старт был стабильным.
        nn.init.zeros_(self.l_head.weight)
        nn.init.zeros_(self.u_off_head.weight)
        nn.init.zeros_(self.u_diag_head.weight)
        nn.init.zeros_(self.l_head.bias)
        nn.init.zeros_(self.u_off_head.bias)
        nn.init.zeros_(self.u_diag_head.bias)

    def _vector_to_lower_unit(self, vec: torch.Tensor) -> torch.Tensor:
        batch_size = vec.shape[0]
        l = torch.eye(self.n, dtype=vec.dtype, device=vec.device).unsqueeze(0).repeat(batch_size, 1, 1)
        idx = torch.tril_indices(row=self.n, col=self.n, offset=-1, device=vec.device)
        l[:, idx[0], idx[1]] = vec
        return l

    def _vector_to_upper(self, off_diag: torch.Tensor, diag_raw: torch.Tensor) -> torch.Tensor:
        batch_size = off_diag.shape[0]
        u = torch.zeros((batch_size, self.n, self.n), dtype=off_diag.dtype, device=off_diag.device)
        idx = torch.triu_indices(row=self.n, col=self.n, offset=1, device=off_diag.device)
        u[:, idx[0], idx[1]] = off_diag
        diag = F.softplus(diag_raw) + 1e-3
        d_idx = torch.arange(self.n, device=off_diag.device)
        u[:, d_idx, d_idx] = diag
        return u

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
        h = self.input_proj(x)

        for block in self.blocks:
            h = h + block(h)

        l = self._vector_to_lower_unit(self.l_head(h))
        u = self._vector_to_upper(self.u_off_head(h), self.u_diag_head(h))
        t = l @ u
        return t
