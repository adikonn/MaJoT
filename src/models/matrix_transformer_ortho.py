"""MatrixTransformerOrtho: MatrixTransformer с гарантированно ортогональным выходом.

Единственное отличие от MatrixTransformer: вместо T = I + δ финальный слой
строит кососимметричную матрицу и берёт от неё матричную экспоненту:

    raw  = output_proj(x)           # (batch, n, n)
    skew = raw - raw^T              # кососимметричная => matrix_exp(skew) ∈ O(n)
    T    = matrix_exp(skew)

При нулевой инициализации output_proj получаем skew = 0, matrix_exp(0) = I —
тот же безопасный старт, что и у базовой модели.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Triangularizer


class MatrixTransformerOrtho(nn.Module, Triangularizer):
    name = "matrix_transformer_ortho"

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_n: int = 32,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_n = max_n

        self.input_proj = nn.Linear(2, hidden_dim)
        self.row_embed = nn.Embedding(max_n, hidden_dim)
        self.col_embed = nn.Embedding(max_n, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

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

        device = A.device

        x = torch.stack([A, B], dim=-1).reshape(batch, n * n, 2)
        x = self.input_proj(x)

        i_idx = torch.arange(n, device=device).unsqueeze(1).expand(n, n).reshape(-1)
        j_idx = torch.arange(n, device=device).unsqueeze(0).expand(n, n).reshape(-1)
        pos = self.row_embed(i_idx) + self.col_embed(j_idx)
        x = x + pos.unsqueeze(0)

        x = self.transformer(x)

        raw = self.output_proj(x).reshape(batch, n, n)
        skew = raw - raw.transpose(-1, -2)
        T = torch.linalg.matrix_exp(skew)

        if squeeze:
            T = T.squeeze(0)
        return T

    @torch.no_grad()
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(A, B)
