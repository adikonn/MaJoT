"""DualStreamRowCol: раздельные энкодеры строк/столбцов с cross-attention для совместной триангуляции.

Идея архитектуры:
    У матрицы есть естественная структура строк и столбцов. Вместо одной длинной
    последовательности из n^2 токенов (как в MatrixTransformer) кодируем каждую строку
    и каждый столбец как короткую последовательность пар (A_ij, B_ij), применяем
    self-attention вдоль ортогональной оси, затем cross-attention между осями и
    уточняющий transformer на сетке n×n. Выход: T = I + δ, на старте T = I.

Почему подходит задаче:
    Совместная триангуляция — глобальное согласованное преобразование; виды «по строкам»
    и «по столбцам» дают дополняющий контекст. Cross-attention связывает оси за O(n)
    токенов на ось, а не за O(n^2).

Допущения:
    n <= max_n (таблицы позиционных эмбеддингов). T не ограничена O(n) на инференсе;
    ортогональность задаётся только штрафом L_orth при обучении.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import Triangularizer


def _make_encoder_layer(
    hidden_dim: int,
    num_heads: int,
    ffn_mult: int,
    dropout: float,
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * ffn_mult,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )


class DualStreamRowCol(nn.Module, Triangularizer):
    """Предсказывает T по паре (A, B) за один forward pass."""

    name = "dual_stream_rowcol"

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
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

        row_layer = _make_encoder_layer(hidden_dim, num_heads, ffn_mult, dropout)
        col_layer = _make_encoder_layer(hidden_dim, num_heads, ffn_mult, dropout)
        self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=num_layers)
        self.col_encoder = nn.TransformerEncoder(col_layer, num_layers=num_layers)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        refine_layers = max(1, num_layers // 2)
        refine_layer = _make_encoder_layer(hidden_dim, num_heads, ffn_mult, dropout)
        self.grid_refine = nn.TransformerEncoder(refine_layer, num_layers=refine_layers)

        # Взвешенное усреднение вдоль оси (вместо простого mean) — больше выразительности.
        self.row_pool = nn.Linear(hidden_dim, 1)
        self.col_pool = nn.Linear(hidden_dim, 1)

        # Нулевая инициализация головы => δ = 0 => T = I в начале обучения.
        self.delta_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def _attention_pool(self, seq: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        """seq: (B*, L, H) -> (B*, H)."""
        weights = torch.softmax(scorer(seq).squeeze(-1), dim=-1)
        return (seq * weights.unsqueeze(-1)).sum(dim=1)

    def _positional_features(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Признаки по ячейкам: (batch, n, n, hidden_dim)."""
        batch, n, _ = A.shape
        if n > self.max_n:
            raise ValueError(f"n={n} превышает max_n={self.max_n}")

        device = A.device
        pair = torch.stack([A, B], dim=-1)
        h = self.input_proj(pair)

        i_idx = torch.arange(n, device=device)
        j_idx = torch.arange(n, device=device)
        h = h + self.row_embed(i_idx).unsqueeze(0).unsqueeze(2)
        h = h + self.col_embed(j_idx).unsqueeze(0).unsqueeze(1)
        return h

    def _encode_rows(self, h: torch.Tensor) -> torch.Tensor:
        """Self-attention вдоль столбцов внутри каждой строки -> (batch, n, H)."""
        batch, n, _, hidden = h.shape
        seq = h.reshape(batch * n, n, hidden)
        out = self.row_encoder(seq)
        pooled = self._attention_pool(out, self.row_pool)
        return pooled.reshape(batch, n, hidden)

    def _encode_cols(self, h: torch.Tensor) -> torch.Tensor:
        """Self-attention вдоль строк внутри каждого столбца -> (batch, n, H)."""
        batch, n, _, hidden = h.shape
        seq = h.permute(0, 2, 1, 3).reshape(batch * n, n, hidden)
        out = self.col_encoder(seq)
        pooled = self._attention_pool(out, self.col_pool)
        return pooled.reshape(batch, n, hidden)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A, B: (batch, n, n) или (n, n). Возвращает T той же формы."""
        squeeze = False
        if A.dim() == 2:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)
            squeeze = True

        batch, n, _ = A.shape
        h = self._positional_features(A, B)
        row_h = self._encode_rows(h)
        col_h = self._encode_cols(h)

        row_ctx, _ = self.cross_attn(row_h, col_h, col_h, need_weights=False)
        row_ctx = self.cross_norm(row_h + row_ctx)

        col_ctx, _ = self.cross_attn(col_h, row_h, row_h, need_weights=False)
        col_ctx = self.cross_norm(col_h + col_ctx)

        fused = row_ctx.unsqueeze(2) + col_ctx.unsqueeze(1)
        fused = fused / math.sqrt(2.0)
        grid = fused.reshape(batch, n * n, self.hidden_dim)
        grid = self.grid_refine(grid)
        delta = self.delta_head(grid).reshape(batch, n, n)

        eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(batch, n, n)
        T = eye + delta

        if squeeze:
            T = T.squeeze(0)
        return T

    @torch.no_grad()
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(A, B)
