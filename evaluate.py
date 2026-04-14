import numpy as np
import torch

from train import load, inference


# Функция для подсчета нашей метрики (сам придумал почти)
def calc_lower_diag_metric(matrix: np.ndarray) -> float:
    lower_triangular = np.tril(matrix, k=-1)
    return float(np.abs(lower_triangular).sum())


def main():
    model = load()
    data = np.load("dataset.npz")
    total_metric = 0
    triang_metric = 0
    triang_cnt = 0
    for i in range(len(data['A'])):
        A = torch.tensor(data['A'][i], dtype=torch.float32)
        B = torch.tensor(data['B'][i], dtype=torch.float32)
        triang = data['y'][i]
        T = inference(model, A, B)
        T_inv = torch.inverse(T)
        M_A = (T_inv @ A @ T).detach().cpu().numpy()
        M_B = (T_inv @ B @ T).detach().cpu().numpy()

        metric_A = calc_lower_diag_metric(M_A)
        metric_B = calc_lower_diag_metric(M_B)
        total_metric += (metric_A + metric_B)
        if (triang == 1):
            triang_cnt += 1
            triang_metric += (metric_A + metric_B)

    print(f"General Score: {((1 / len(data['A'])) * total_metric):.4f}")
    print(f"Score for triang matrices: {((1 / triang_cnt) * triang_metric):.4f}")


if __name__ == "__main__":
    main()
