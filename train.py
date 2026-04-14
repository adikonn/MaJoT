import torch
import torch.nn as nn
import torch.optim as optim


def normalize_matrix(M):
    return M / (torch.norm(M) + 1e-8)


def safe_matrix_exp(M, scale=0.1):
    return torch.matrix_exp(scale * M)


def generate_matrices(n):
    A = torch.randn(n, n)
    B = torch.randn(n, n)
    return A, B


def lower_triangle_mask(n):
    mask = torch.tril(torch.ones(n, n), diagonal=-1)
    return mask


class Triangularizer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.M = nn.Parameter(torch.randn(n, n) * 0.1)

    def forward(self, A, B):
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        if not isinstance(B, torch.Tensor):
            B = torch.tensor(B, dtype=torch.float32, requires_grad=False)
        T = torch.matrix_exp(0.1 * self.M)
        T_inv = torch.inverse(T)

        A_t = T_inv @ A @ T
        B_t = T_inv @ B @ T

        return A_t, B_t, T


def loss_function(A_t, B_t, mask):
    def normalized_lower(M):
        lower = (M * mask)
        return torch.sum(lower ** 2) / (torch.sum(M ** 2) + 1e-8)

    return normalized_lower(A_t) + normalized_lower(B_t)


def triangular_score(M):
    n = M.shape[0]
    mask = torch.tril(torch.ones(n, n), diagonal=-1)
    total = torch.sum(M ** 2)
    lower = torch.sum((M * mask) ** 2)

    return 1.0 - (lower / total + 1e-8)


def train(n=5, epochs=2000, lr=1e-2):
    A, B = generate_matrices(n)

    A = normalize_matrix(A)
    B = normalize_matrix(B)

    mask = lower_triangle_mask(n)

    model = Triangularizer(n)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        A_t, B_t, T = model(A, B)
        loss = loss_function(A_t, B_t, mask)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if epoch % 200 == 0:
            score_A = triangular_score(A_t).item()
            score_B = triangular_score(B_t).item()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"ScoreA={score_A:.4f}, ScoreB={score_B:.4f}")

    return model


def inference(model, A, B):
    A_t, B_t, T = model(A, B)
    return T


def save(model, filename='triangularizer.pt'):
    torch.save(model.state_dict(), filename)


def load(filename='triangularizer.pt', n=3):
    model = Triangularizer(n)  # Создать новую модель
    model.load_state_dict(torch.load(filename, weights_only=True))  # Безопасно!
    model.eval()
    return model


if __name__ == "__main__":
    N = 3
    model = train(n=N, epochs=12345)
    save(model)
