import os
import torch


MATRIX_TYPES = ("perfect", "noisy", "random")
DEFAULT_NOISE_LEVEL = 1e-3


def generate_perfect(n, dtype=torch.float32):
    # Генерируем случайную ортогональную матрицу Q через QR-разложение
    H = torch.randn(n, n, dtype=dtype)
    Q, R = torch.linalg.qr(H)
    Q = Q * torch.sign(torch.diag(R))   # теперь Q ~ Haar на O(n)

    # Случайные верхнетреугольные матрицы
    T_A = torch.triu(torch.randn(n, n, dtype=dtype))
    T_B = torch.triu(torch.randn(n, n, dtype=dtype))

    # A = Q * T_A * Q^T
    matrix_a = Q @ T_A @ Q.T
    matrix_b = Q @ T_B @ Q.T
    return matrix_a, matrix_b


def generate_noisy(n, noise_level=1e-3, dtype=torch.float32):
    matrix_a, matrix_b = generate_perfect(n, dtype=dtype)
    matrix_a += noise_level * matrix_a.std() * torch.randn(n, n, dtype=dtype)
    matrix_b += noise_level * matrix_b.std() * torch.randn(n, n, dtype=dtype)
    return matrix_a, matrix_b


def generate_random(n, dtype=torch.float32):
    matrix_a = torch.randn(n, n, dtype=dtype)
    matrix_b = torch.randn(n, n, dtype=dtype)
    return matrix_a, matrix_b



def generate_synthetic_pair(matrix_type, size, noise_level=DEFAULT_NOISE_LEVEL, dtype=torch.float32):
    if matrix_type == "perfect":
        return generate_perfect(size, dtype=dtype)
    if matrix_type == "noisy":
        return generate_noisy(size, noise_level=noise_level, dtype=dtype)
    if matrix_type == "random":
        return generate_random(size, dtype=dtype)
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


def create_and_save_dataset(save_path, sizes, samples_per_config=20, noise_level=DEFAULT_NOISE_LEVEL, seed=42):
    torch.manual_seed(seed)

    dataset = []

    for n in sizes:
        for _ in range(samples_per_config):
            for matrix_type in MATRIX_TYPES:
                A, B = generate_synthetic_pair(matrix_type, size=n, noise_level=noise_level)
                dataset.append({"n": n, "type": matrix_type, "A": A, "B": B})
            
    torch.save(dataset, save_path)
    print(f"Dataset generated and saved to {save_path}")
    print(f"Total samples: {len(dataset)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "dataset.pt")

    sizes_to_test = [4, 8]
    create_and_save_dataset(save_path, sizes=sizes_to_test, samples_per_config=5)
