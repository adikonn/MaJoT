import numpy as np
import joblib

# Функция для подсчета нашей метрики (сам придумал почти)
def calc_lower_diag_metric(matrix: np.ndarray) -> float:
    lower_triangular = np.tril(matrix, k=-1)
    return float(np.abs(lower_triangular).sum())


def main():
    model = joblib.load('model.joblib')
    data = np.load("dataset.npz")
    total_metric = 0
    for i in range(len(data['A'])):
        A = data['A'][i]
        B = data['B'][i]

        T = model.predict(A, B)

        M_A = T.T @ A @ T
        M_B = T.T @ B @ T

        metric_A = calc_lower_diag_metric(M_A)
        metric_B = calc_lower_diag_metric(M_B)
        total_metric += (metric_A + metric_B)

    print(f"Your Score: {total_metric:.4f}")