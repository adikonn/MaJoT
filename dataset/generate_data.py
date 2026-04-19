import numpy as np
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from classical_methods import check_pair_triangularizable


def generate_jointly_triangularizable_pair(n=3):
    Ta = np.triu(np.random.randn(n, n))
    Tb = np.triu(np.random.randn(n, n))

    P = np.random.randn(n, n)
    while abs(np.linalg.det(P)) < 1e-2:
        P = np.random.randn(n, n)

    P_inv = np.linalg.inv(P)

    A = P @ Ta @ P_inv
    B = P @ Tb @ P_inv

    return A, B


def generate_random_pair(n=3):
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    return A, B


def create_dataset(n_joint=100, n_random=150, n_size=3, seed=42, ensure_hard_negatives=True):
    np.random.seed(seed)

    data_A = []
    data_B = []
    labels =[]
    for _ in range(n_joint):
        A, B = generate_jointly_triangularizable_pair(n=n_size)
        data_A.append(A)
        data_B.append(B)
        labels.append(1)

    neg_added = 0
    max_attempts = max(10 * n_random, 100)
    attempts = 0
    while neg_added < n_random and attempts < max_attempts:
        attempts += 1
        A, B = generate_random_pair(n=n_size)
        if ensure_hard_negatives and check_pair_triangularizable(A, B):
            continue
        data_A.append(A)
        data_B.append(B)
        labels.append(0)
        neg_added += 1

    if neg_added < n_random:
        raise RuntimeError(
            f"Failed to generate enough non-triangularizable negatives: got {neg_added}/{n_random}."
        )

    data_A = np.array(data_A)
    data_B = np.array(data_B)
    labels = np.array(labels)

    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    data_A = data_A[indices]
    data_B = data_B[indices]

    return data_A, data_B, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for joint triangularization.")
    parser.add_argument("--n-joint", type=int, default=300)
    parser.add_argument("--n-random", type=int, default=300)
    parser.add_argument("--n-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-noisy-negatives",
        action="store_true",
        help="If set, random class is not filtered by classical criterion.",
    )
    args = parser.parse_args()

    print("Генерация датасета...")
    X_A, X_B, labels = create_dataset(
        n_joint=args.n_joint,
        n_random=args.n_random,
        n_size=args.n_size,
        seed=args.seed,
        ensure_hard_negatives=not args.allow_noisy_negatives,
    )

    print("\nРазмеры полученных тензоров:")
    print(f"Матрицы A (X_A): {X_A.shape}")  # (250, 3, 3)
    print(f"Матрицы B (X_B): {X_B.shape}")  # (250, 3, 3)

    os.makedirs("dataset", exist_ok=True)
    filename = os.path.join("dataset", "dataset.npz")
    np.savez(filename, A=X_A, B=X_B, y=labels)
    print(f"\nДатасет успешно сохранен в файл: {filename}")
