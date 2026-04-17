import numpy as np


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


def create_dataset(n_joint=100, n_random=150, n_size=3, seed=42):
    np.random.seed(seed)

    data_A = []
    data_B = []
    labels =[]
    for _ in range(n_joint):
        A, B = generate_jointly_triangularizable_pair(n=n_size)
        data_A.append(A)
        data_B.append(B)
        labels.append(1)

    for _ in range(n_random):
        A, B = generate_random_pair(n=n_size)
        data_A.append(A)
        data_B.append(B)
        labels.append(0)

    data_A = np.array(data_A)
    data_B = np.array(data_B)
    labels = np.array(labels)

    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    data_A = data_A[indices]
    data_B = data_B[indices]

    return data_A, data_B, labels


if __name__ == "__main__":
    print("Генерация датасета...")
    X_A, X_B, labels = create_dataset(n_joint=300, n_random=300, n_size=3)

    print("\nРазмеры полученных тензоров:")
    print(f"Матрицы A (X_A): {X_A.shape}")  # (250, 3, 3)
    print(f"Матрицы B (X_B): {X_B.shape}")  # (250, 3, 3)

    filename = "dataset.npz"
    np.savez(filename, A=X_A, B=X_B, y=labels)
    print(f"\nДатасет успешно сохранен в файл: {filename}")
