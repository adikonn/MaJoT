import torch
from torch import nn


class CrossAttnTriangularizer(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj_a = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj_b = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.out_net[2].weight)
        nn.init.zeros_(self.out_net[2].bias)

    def forward(self, A, B):
        is_unbatched = A.dim() == 2
        if is_unbatched:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)

        b_size, n, _ = A.shape
        device = A.device

        coords = torch.linspace(-1, 1, steps=n, device=device)
        grid_i, grid_j = torch.meshgrid(coords, coords, indexing="ij")

        grid_i = grid_i.reshape(1, n * n, 1).expand(b_size, -1, -1)
        grid_j = grid_j.reshape(1, n * n, 1).expand(b_size, -1, -1)

        flat_a = A.reshape(b_size, n * n, 1)
        flat_b = B.reshape(b_size, n * n, 1)

        in_a = torch.cat([flat_a, grid_i, grid_j], dim=-1)
        in_b = torch.cat([flat_b, grid_i, grid_j], dim=-1)

        feat_a = self.proj_a(in_a)
        feat_b = self.proj_b(in_b)

        attn_ab, _ = self.cross_attn(query=feat_a, key=feat_b, value=feat_b)
        attn_ba, _ = self.cross_attn(query=feat_b, key=feat_a, value=feat_a)

        x = self.norm(feat_a + feat_b + attn_ab + attn_ba)

        out = self.out_net(x).reshape(b_size, n, n)

        I = torch.eye(n, device=device).unsqueeze(0).expand(b_size, -1, -1)
        T = out + I

        if is_unbatched:
            T = T.squeeze(0)

        return T

    def find_transform(self, A, B):
        with torch.no_grad():
            return self.forward(A, B)
