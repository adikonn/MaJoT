"""MatrixTransformer: a transformer over matrix entries.

Tokenization:
    Each element position (i, j) of the input pair of matrices A, B becomes a token
    whose initial features are (A[i, j], B[i, j]) augmented with learned row and
    column positional embeddings.

Output:
    The model produces a residual `delta` of shape (n, n) and returns
        T = I + delta
    so that at initialization T is the identity (a trivial but stable starting point).

This design handles variable matrix sizes naturally: the only architectural limit is
`max_n`, which controls the size of the positional embedding tables.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Triangularizer


class MatrixTransformer(nn.Module, Triangularizer):
    name = "matrix_transformer"

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

        # 2 input features per (i, j): A[i, j] and B[i, j]
        self.input_proj = nn.Linear(2, hidden_dim)

        # Learned positional embeddings for row and column indices.
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

        # Output close to zero at init -> T == I at the start of training.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A, B: (batch, n, n) or (n, n). Returns T of the same batch shape."""
        squeeze = False
        if A.dim() == 2:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)
            squeeze = True

        batch, n, _ = A.shape
        if n > self.max_n:
            raise ValueError(f"n={n} exceeds configured max_n={self.max_n}")

        device = A.device

        # (batch, n, n, 2) -> (batch, n*n, 2)
        x = torch.stack([A, B], dim=-1).reshape(batch, n * n, 2)
        x = self.input_proj(x)

        # 2D positional encoding.
        i_idx = torch.arange(n, device=device).unsqueeze(1).expand(n, n).reshape(-1)
        j_idx = torch.arange(n, device=device).unsqueeze(0).expand(n, n).reshape(-1)
        pos = self.row_embed(i_idx) + self.col_embed(j_idx)  # (n*n, hidden_dim)
        x = x + pos.unsqueeze(0)

        x = self.transformer(x)  # (batch, n*n, hidden_dim)

        delta = self.output_proj(x).reshape(batch, n, n)  # residual

        eye = torch.eye(n, device=device, dtype=A.dtype).expand(batch, n, n)
        T = eye + delta

        if squeeze:
            T = T.squeeze(0)
        return T

    # ----- Triangularizer interface -----
    @torch.no_grad()
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(A, B)
