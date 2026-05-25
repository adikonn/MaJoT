"""IterativeRefinementOrtho: итеративное уточнение с ортогональными шагами.

Отличие от IterativeRefinementTriangularizer: на каждом шаге поправка
применяется как правый множитель из O(n), а не как аддитивный сдвиг:

    raw   = delta_head(h_fused)          # (batch, n, n)
    skew  = raw - raw^T                  # кососимметричная
    R_k   = matrix_exp(alpha_k * skew)   # R_k ∈ O(n)
    T     = T @ R_k                       # T остаётся ортогональной на каждом шаге

При нулевой инициализации delta_head: skew = 0, R_k = I, T не меняется.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Triangularizer


class IterativeRefinementOrtho(nn.Module, Triangularizer):
    """Предсказывает ортогональную T через K итераций с весовым разделением."""

    name = "iterative_refinement_ortho"

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_steps: int = 6,
        max_n: int = 32,
        dropout: float = 0.0,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.max_n = max_n

        self.feat_proj = nn.Linear(3, hidden_dim)

        self.row_embed = nn.Embedding(max_n, hidden_dim)
        self.col_embed = nn.Embedding(max_n, hidden_dim)

        row_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=1)

        col_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.col_encoder = nn.TransformerEncoder(col_layer, num_layers=1)

        self.delta_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        # log-параметризация для позитивности; init ≈ 0.135
        self.log_step_sizes = nn.Parameter(torch.full((num_steps,), -2.0))

    def _compute_rotation(
        self, T: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """Compute one orthogonal correction R ∈ O(n) from current T and residuals."""
        batch, n, _ = T.shape
        device = T.device

        M_A = T.transpose(-2, -1) @ A @ T
        M_B = T.transpose(-2, -1) @ B @ T
        R_A = torch.tril(M_A, diagonal=-1)
        R_B = torch.tril(M_B, diagonal=-1)

        feat = torch.stack([T, R_A, R_B], dim=-1)
        h = self.feat_proj(feat)

        i_idx = torch.arange(n, device=device)
        j_idx = torch.arange(n, device=device)
        h = h + self.row_embed(i_idx).view(1, n, 1, self.hidden_dim)
        h = h + self.col_embed(j_idx).view(1, 1, n, self.hidden_dim)

        h_row = self.row_encoder(
            h.reshape(batch * n, n, self.hidden_dim)
        ).reshape(batch, n, n, self.hidden_dim)

        h_col = self.col_encoder(
            h.permute(0, 2, 1, 3).reshape(batch * n, n, self.hidden_dim)
        ).reshape(batch, n, n, self.hidden_dim).permute(0, 2, 1, 3)

        h_fused = (h_row + h_col) * 0.5
        raw = self.delta_head(h_fused).squeeze(-1)  # (batch, n, n)

        skew = raw - raw.transpose(-1, -2)
        return skew

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A, B: (batch, n, n) or (n, n). Returns orthogonal T of the same batch shape."""
        squeeze = False
        if A.dim() == 2:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)
            squeeze = True

        batch, n, _ = A.shape
        if n > self.max_n:
            raise ValueError(f"n={n} exceeds configured max_n={self.max_n}")

        T = torch.eye(n, dtype=A.dtype, device=A.device).unsqueeze(0).expand(batch, -1, -1).clone()
        step_sizes = torch.exp(self.log_step_sizes)

        for k in range(self.num_steps):
            skew = self._compute_rotation(T, A, B)
            # Каждый шаг — умножение на ортогональную матрицу из O(n)
            R_k = torch.linalg.matrix_exp(step_sizes[k] * skew)
            T = T @ R_k

        if squeeze:
            T = T.squeeze(0)
        return T

    @torch.no_grad()
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(A, B)
