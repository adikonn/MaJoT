"""LearnedGivens: learn a joint triangularizer via a learned product of Givens rotations."""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import Triangularizer


def _as_batched(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    squeeze = False
    if A.dim() == 2:
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        squeeze = True
    return A, B, squeeze


class LearnedGivens(nn.Module, Triangularizer):
    """Predicts an orthogonal T as a product of learned Givens rotations."""

    name = "learned_givens"

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        max_n: int = 32,
        num_rotations: int = 64,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        self.num_rotations = num_rotations

        self.input_proj = nn.Linear(2, hidden_dim)
        self.row_embed = nn.Embedding(max_n, hidden_dim)
        self.col_embed = nn.Embedding(max_n, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pool = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.head_i = nn.Linear(hidden_dim, num_rotations)
        self.head_j = nn.Linear(hidden_dim, num_rotations)
        self.head_theta = nn.Linear(hidden_dim, num_rotations)

        nn.init.zeros_(self.head_theta.weight)
        nn.init.zeros_(self.head_theta.bias)

    def _encode(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Return a pooled representation (batch, hidden_dim)."""
        batch, n, _ = A.shape
        if n > self.max_n:
            msg = f"n={n} exceeds configured max_n={self.max_n}"
            raise ValueError(msg)

        device = A.device
        x = torch.stack([A, B], dim=-1).reshape(batch, n * n, 2)
        x = self.input_proj(x)

        i_idx = torch.arange(n, device=device).unsqueeze(1).expand(n, n).reshape(-1)
        j_idx = torch.arange(n, device=device).unsqueeze(0).expand(n, n).reshape(-1)
        pos = self.row_embed(i_idx) + self.col_embed(j_idx)
        x = x + pos.unsqueeze(0)

        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.pool(x)

    @staticmethod
    def _apply_givens_left(T: torch.Tensor, i: int, j: int, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        Ti = T[:, i, :].clone()
        Tj = T[:, j, :].clone()
        new_Ti = c[:, None] * Ti - s[:, None] * Tj
        new_Tj = s[:, None] * Ti + c[:, None] * Tj
        T[:, i, :] = new_Ti
        T[:, j, :] = new_Tj
        return T

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A, B, squeeze = _as_batched(A, B)
        batch, n, _ = A.shape

        h = self._encode(A, B)

        i_logits = self.head_i(h)
        j_logits = self.head_j(h)
        theta_raw = self.head_theta(h)

        theta = math.pi * torch.tanh(theta_raw)
        pair_gate = torch.sigmoid(i_logits + j_logits)
        theta = theta * pair_gate

        T = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(batch, n, n).clone()

        pairs = []
        for p in range(n):
            for q in range(p + 1, n):
                pairs.append((p, q))
        if len(pairs) == 0:
            pairs = [(0, 0)]

        for k in range(self.num_rotations):
            i, j = pairs[k % len(pairs)]
            th = theta[:, k]
            c = torch.cos(th)
            s = torch.sin(th)
            T = self._apply_givens_left(T, i, j, c, s)

        if squeeze:
            T = T.squeeze(0)
        return T

    @torch.no_grad()
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(A, B)
