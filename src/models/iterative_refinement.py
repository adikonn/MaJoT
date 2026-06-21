"""IterativeRefinementTriangularizer: unrolled gradient-like steps with row-column attention.

Architecture idea:
    Starts from T=I and takes K learned update steps. At each step, a weight-shared
    row-column attention module processes the current T together with triangularization
    residuals (tril(T^T A T, -1) and tril(T^T B T, -1)) to predict a correction delta.
    The update rule T ← T − α_k · delta mirrors gradient descent on the triangularization
    loss but replaces the true gradient with a learned approximation conditioned on the
    full state (T, R_A, R_B). Per-step sizes α_k are learnable positive scalars.

Why it suits the task:
    All classical baselines (Jacobi, Schur, Newton) are iterative — they reduce the
    lower-triangular residual step by step. This architecture shares that inductive bias
    while learning the update rule from data. Explicit residual feedback allows the model
    to correct its own mistakes across steps, unlike one-shot models (MatrixTransformer,
    DualStreamRowCol) that produce T in a single forward pass without seeing the current
    triangularization quality.

Assumptions:
    n ≤ max_n (positional embedding table limit). Orthogonality of T is encouraged only
    by the λ · L_orth penalty, not enforced structurally. Attention weights are shared
    across all K steps (parameter-efficient unrolled optimisation).
"""
from __future__ import annotations

import torch
from torch import nn

from .base import Triangularizer


class IterativeRefinementTriangularizer(nn.Module, Triangularizer):
    """Predicts T via K iterative refinement steps with a shared row-column attention module."""

    name = "iterative_refinement"

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_steps: int = 6,
        max_n: int = 32,
        dropout: float = 0.0,
        ffn_mult: int = 2,
    ) -> None:
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

        self.log_step_sizes = nn.Parameter(torch.full((num_steps,), -2.0))

    def _compute_delta(
        self, T: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one correction delta from current T and triangularization residuals."""
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
            h.reshape(batch * n, n, self.hidden_dim),
        ).reshape(batch, n, n, self.hidden_dim)

        h_col = self.col_encoder(
            h.permute(0, 2, 1, 3).reshape(batch * n, n, self.hidden_dim),
        ).reshape(batch, n, n, self.hidden_dim).permute(0, 2, 1, 3)

        h_fused = (h_row + h_col) * 0.5
        return self.delta_head(h_fused).squeeze(-1)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A, B: (batch, n, n) or (n, n). Returns T of the same batch shape."""
        squeeze = False
        if A.dim() == 2:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)
            squeeze = True

        batch, n, _ = A.shape
        if n > self.max_n:
            msg = f"n={n} exceeds configured max_n={self.max_n}"
            raise ValueError(msg)

        T = torch.eye(n, dtype=A.dtype, device=A.device).unsqueeze(0).expand(batch, -1, -1).clone()
        step_sizes = torch.exp(self.log_step_sizes)

        for k in range(self.num_steps):
            delta = self._compute_delta(T, A, B)
            T = T - step_sizes[k] * delta

        if squeeze:
            T = T.squeeze(0)
        return T

    @torch.no_grad()
    def find_transform(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(A, B)
