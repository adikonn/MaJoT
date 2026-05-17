import os
import torch


MATRIX_TYPES = ("perfect", "noisy", "random")


def generate_perfect(n, dtype=torch.float32):
    # Генерируем случайную ортогональную матрицу Q через QR-разложение
    H = torch.randn(n, n, dtype=dtype)
    Q, _ = torch.linalg.qr(H)
    
    # Случайные верхнетреугольные матрицы
    T_A = torch.triu(torch.randn(n, n, dtype=dtype))
    T_B = torch.triu(torch.randn(n, n, dtype=dtype))
    
    # A = Q * T_A * Q^T
    A = Q @ T_A @ Q.T
    B = Q @ T_B @ Q.T
    return A, B

def generate_noisy(n, noise_level=1e-3, dtype=torch.float32):
    A, B = generate_perfect(n, dtype=dtype)
    A += noise_level * torch.randn(n, n, dtype=dtype)
    B += noise_level * torch.randn(n, n, dtype=dtype)
    return A, B

def generate_random(n, dtype=torch.float32):
    A = torch.randn(n, n, dtype=dtype)
    B = torch.randn(n, n, dtype=dtype)
    return A, B


def generate_synthetic_pair(matrix_type, n, dtype=torch.float32):
    generators = {
        "perfect": generate_perfect,
        "noisy": generate_noisy,
        "random": generate_random,
    }
    if matrix_type not in generators:
        raise ValueError(f"Unknown matrix type: {matrix_type}")
    return generators[matrix_type](n, dtype=dtype)


def create_and_save_dataset(save_path, sizes, samples_per_config=20, seed=42):
    torch.manual_seed(seed)
    
    dataset = []
    
    for n in sizes:
        for _ in range(samples_per_config):
            for matrix_type in MATRIX_TYPES:
                A, B = generate_synthetic_pair(matrix_type, n)
                dataset.append({"n": n, "type": matrix_type, "A": A, "B": B})
            
    torch.save(dataset, save_path)
    print(f"Dataset generated and saved to {save_path}")
    print(f"Total samples: {len(dataset)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "dataset.pt")
    
    sizes_to_test = [4, 8]
    create_and_save_dataset(save_path, sizes=sizes_to_test, samples_per_config=5)
