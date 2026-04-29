import argparse
from pathlib import Path

import numpy as np


def _sample_invertible_matrix(n: int, rng: np.random.Generator, max_cond: float = 50.0) -> np.ndarray:
    while True:
        p = rng.normal(size=(n, n))
        det = np.linalg.det(p)
        if abs(det) < 1e-4:
            continue
        cond = np.linalg.cond(p)
        if np.isfinite(cond) and cond <= max_cond:
            return p


def _sample_upper_triangular(n: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    t = np.triu(rng.normal(loc=0.0, scale=scale, size=(n, n)))
    diag_noise = rng.normal(loc=0.0, scale=0.25 * scale, size=n)
    np.fill_diagonal(t, np.diag(t) + diag_noise)
    return t


def generate_jointly_triangularizable_pair(n: int, rng: np.random.Generator, scale: float) -> tuple[np.ndarray, np.ndarray]:
    ta = _sample_upper_triangular(n=n, rng=rng, scale=scale)
    tb = _sample_upper_triangular(n=n, rng=rng, scale=scale)

    p = _sample_invertible_matrix(n=n, rng=rng)
    p_inv = np.linalg.inv(p)

    a = p @ ta @ p_inv
    b = p @ tb @ p_inv
    return a, b


def _rotation_block(theta: float, radius: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return radius * np.array([[c, -s], [s, c]], dtype=np.float64)


def _matrix_with_complex_eigs(n: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    base = np.zeros((n, n), dtype=np.float64)
    theta = float(rng.uniform(0.2, 1.2))
    radius = float(rng.uniform(0.5, 1.5) * scale)
    base[:2, :2] = _rotation_block(theta=theta, radius=radius)

    if n > 2:
        diag = rng.normal(loc=0.0, scale=scale, size=n - 2)
        base[2:, 2:] = np.diag(diag)

    p = _sample_invertible_matrix(n=n, rng=rng)
    p_inv = np.linalg.inv(p)
    return p @ base @ p_inv


def generate_non_joint_pair(n: int, rng: np.random.Generator, scale: float) -> tuple[np.ndarray, np.ndarray]:
    # A имеет комплексно-сопряженные собственные значения => не приводится к верхнетреугольному виду
    # над R, значит и совместной триангуляризации над R для пары не будет.
    a = _matrix_with_complex_eigs(n=n, rng=rng, scale=scale)
    b = rng.normal(loc=0.0, scale=scale, size=(n, n))
    return a, b


def create_dataset(
    n_joint: int = 50_000,
    n_non_joint: int = 50_000,
    n_size: int = 3,
    seed: int = 42,
    scales: tuple[float, ...] = (0.25, 1.0, 4.0, 10.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_size < 2:
        raise ValueError("n_size must be >= 2")

    rng = np.random.default_rng(seed)

    total = n_joint + n_non_joint
    data_a = np.empty((total, n_size, n_size), dtype=np.float64)
    data_b = np.empty((total, n_size, n_size), dtype=np.float64)
    labels = np.empty((total,), dtype=np.int64)

    for i in range(n_joint):
        scale = float(rng.choice(scales))
        a, b = generate_jointly_triangularizable_pair(n=n_size, rng=rng, scale=scale)
        data_a[i] = a
        data_b[i] = b
        labels[i] = 1

    for i in range(n_non_joint):
        scale = float(rng.choice(scales))
        a, b = generate_non_joint_pair(n=n_size, rng=rng, scale=scale)
        j = n_joint + i
        data_a[j] = a
        data_b[j] = b
        labels[j] = 0

    indices = rng.permutation(total)
    data_a = data_a[indices]
    data_b = data_b[indices]
    labels = labels[indices]
    return data_a, data_b, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large dataset for joint triangulation task")
    parser.add_argument("--n-size", type=int, default=3, help="Matrix size N")
    parser.add_argument("--n-joint", type=int, default=50_000, help="Number of jointly triangularizable pairs")
    parser.add_argument("--n-non-joint", type=int, default=50_000, help="Number of non-joint pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="dataset/dataset.npz", help="Output .npz path")
    args = parser.parse_args()

    print("Генерация датасета...")
    x_a, x_b, labels = create_dataset(
        n_joint=args.n_joint,
        n_non_joint=args.n_non_joint,
        n_size=args.n_size,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, A=x_a, B=x_b, y=labels)

    print("\nРазмеры полученных тензоров:")
    print(f"Матрицы A (X_A): {x_a.shape}")
    print(f"Матрицы B (X_B): {x_b.shape}")
    print(f"Метки y: {labels.shape}")
    print(f"\nДатасет успешно сохранен в файл: {out_path}")


if __name__ == "__main__":
    main()
