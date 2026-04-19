import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm

from classical_methods import check_pair_triangularizable

try:
    from model.model import TriangularizerModel
except ImportError:
    print("❌ ОШИБКА: Не найден файл model.py или класс TriangularizerModel")
    sys.exit(1)


def calc_lower_diag_metric_batch(M: torch.Tensor) -> torch.Tensor:
    """
    Считает сумму абсолютных значений под главной диагональю
    ДЛЯ КАЖДОЙ матрицы в батче отдельно.
    Возвращает 1D тензор размерности (Batch,).
    """
    lower_triangular = torch.tril(M, diagonal=-1)
    # Суммируем по строкам и столбцам (dim 1 и 2), оставляя размерность батча
    return torch.abs(lower_triangular).sum(dim=(1, 2))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    acc = (tp + tn) / total
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _select_nn_threshold(scores: np.ndarray, labels: np.ndarray, q_min: float = 0.05, q_max: float = 0.95) -> float:
    lo = float(np.quantile(scores, q_min))
    hi = float(np.quantile(scores, q_max))
    if hi <= lo:
        return float(np.median(scores))
    candidates = np.linspace(lo, hi, 120)
    best_thr = candidates[0]
    best_f1 = -1.0
    for thr in candidates:
        pred = (scores <= thr).astype(np.int32)
        f1 = _binary_metrics(labels, pred)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate neural and classical triangularization quality.")
    parser.add_argument("--dataset", default="dataset/dataset.npz", help="Path to dataset NPZ file.")
    parser.add_argument("--weights", default="model_weights.pt", help="Path to trained model weights.")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--calib-ratio", type=float, default=0.25, help="Ratio used for threshold calibration split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for calibration split.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print(f"❌ ОШИБКА: Тестовый датасет {dataset_path} не найден!")
        sys.exit(1)

    data = np.load(dataset_path)
    A_data, B_data = data['A'], data['B']

    y_data = data['y']
    n_samples, n_size, _ = A_data.shape

    model = TriangularizerModel(n=n_size)

    # ВАЖНО: Принудительно переводим все параметры загруженной модели во float64
    model = model.to(torch.float64)

    weights_path = args.weights
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"[OK] Веса модели ({weights_path}) успешно загружены.")
    else:
        print(f"[WARN] Файл {weights_path} не найден. Тестируем случайные веса!")

    model.eval()

    with torch.no_grad():
        A_input = torch.tensor(A_data, dtype=torch.float64)
        B_input = torch.tensor(B_data, dtype=torch.float64)

        batch_size = args.batch_size
        total_scores_list = []

        for i in tqdm(range(0, n_samples, batch_size), desc="Оценка модели (Evaluation)", unit="batch"):
            A_batch = A_input[i:i + batch_size]
            B_batch = B_input[i:i + batch_size]

            T_batch = model(A_batch, B_batch)

            if T_batch.shape != A_batch.shape:
                print(f"\n❌ ОШИБКА: Модель вернула T с размерностью {T_batch.shape}, "
                      f"а ожидалось {A_batch.shape}")
                sys.exit(1)

            if T_batch.dtype != torch.float64:
                print(f"\n❌ ОШИБКА : Модель вернула T с типом {T_batch.dtype}, "
                      f"а ожидался строго torch.float64 (Double). "
                      f"Проверьте, что в forward() вы не создаете тензоры через torch.tensor() без указания dtype.")
                sys.exit(1)
            T_t_batch = torch.transpose(T_batch, dim0=1, dim1=2)
            A_trans = T_t_batch @ A_batch @ T_batch
            B_trans = T_t_batch @ B_batch @ T_batch

            scores_A = calc_lower_diag_metric_batch(A_trans)
            scores_B = calc_lower_diag_metric_batch(B_trans)
            total_scores_list.append(scores_A + scores_B)

        total_scores = torch.cat(total_scores_list, dim=0)

    y_tensor = torch.tensor(y_data, dtype=torch.int32)
    y_np = y_data.astype(np.int32)
    classical_preds = np.array(
        [check_pair_triangularizable(A_data[i], B_data[i]) for i in range(n_samples)],
        dtype=np.int32,
    )

    avg_total_score = total_scores.mean().item()

    triang_mask = (y_tensor == 1)
    triang_count = triang_mask.sum().item()
    if triang_count > 0:
        avg_triang_score = total_scores[triang_mask].mean().item()
        min_triang_score = total_scores[triang_mask].min().item()
        max_triang_score = total_scores[triang_mask].max().item()
    else:
        avg_triang_score = "nan"
        min_triang_score = "nan"
        max_triang_score = "nan"

    print("\n" + "=" * 40)
    print(f"GENERAL_SCORE={avg_total_score:.6f}")
    print(f"TRIANG_SCORE={avg_triang_score:.6f}")
    classical_metrics = _binary_metrics(y_np, classical_preds)
    classical_acc = classical_metrics["acc"]
    nn_scores_np = total_scores.detach().cpu().numpy()
    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(n_samples)
    rng.shuffle(all_idx)
    calib_size = max(1, int(n_samples * args.calib_ratio))
    calib_size = min(calib_size, n_samples - 1)
    calib_idx = all_idx[:calib_size]
    test_idx = all_idx[calib_size:]
    nn_threshold = _select_nn_threshold(nn_scores_np[calib_idx], y_np[calib_idx])
    nn_preds = (nn_scores_np[test_idx] <= nn_threshold).astype(np.int32)
    nn_metrics = _binary_metrics(y_np[test_idx], nn_preds)
    classical_test_metrics = _binary_metrics(y_np[test_idx], classical_preds[test_idx])
    print(f"NN_THRESHOLD={nn_threshold:.6f}")
    print(f"NN_ACC={nn_metrics['acc']:.6f}")
    print(f"NN_F1={nn_metrics['f1']:.6f}")
    print(f"CLASSICAL_ACC={classical_test_metrics['acc']:.6f}")
    print(f"CLASSICAL_F1={classical_test_metrics['f1']:.6f}")
    print("=" * 40 + "\n")

    markdown_report = (
        "| Метрика | Значение |\n"
        "| --- | --- |\n"
        f"| Всего пар в датасете | `{n_samples}` |\n"
        f"| Из них совместно триангуляризуемых | `{triang_count}` |\n"
        f"| **Итоговая ошибка (General Score)** | **`{avg_total_score:.6f}`** |\n"
        f"| **Ошибка на трианг. матрицах (Triang Score)** | **`{avg_triang_score:.6f}`** |\n"
        f"| **Мин. ошибка на трианг. матрицах** | **`{min_triang_score:.6f}`** |\n"
        f"| **Макс. ошибка на трианг. матрицах** | **`{max_triang_score:.6f}`** |\n"
        f"| **Подобранный порог для NN-решения** | **`{nn_threshold:.6f}`** |\n\n"
        f"| Калибровочный размер | `{len(calib_idx)}` |\n"
        f"| Тестовый размер | `{len(test_idx)}` |\n\n"
        "| Алгоритм | Accuracy | Precision | Recall | F1 |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"| NN (по score <= threshold) | `{nn_metrics['acc']:.6f}` | `{nn_metrics['precision']:.6f}` | "
        f"`{nn_metrics['recall']:.6f}` | `{nn_metrics['f1']:.6f}` |\n"
        f"| Classical criterion | `{classical_test_metrics['acc']:.6f}` | `{classical_test_metrics['precision']:.6f}` | "
        f"`{classical_test_metrics['recall']:.6f}` | `{classical_test_metrics['f1']:.6f}` |\n\n"
        "| Confusion Matrix | TP | TN | FP | FN |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"| NN | `{nn_metrics['tp']}` | `{nn_metrics['tn']}` | `{nn_metrics['fp']}` | `{nn_metrics['fn']}` |\n"
        f"| Classical | `{classical_test_metrics['tp']}` | `{classical_test_metrics['tn']}` | "
        f"`{classical_test_metrics['fp']}` | `{classical_test_metrics['fn']}` |\n\n"
    )

    # Сохраняем в файл metrics.md для бота
    with open("metrics.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Дублируем в Summary GitHub Actions
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("### Результаты тестирования (Double Precision)\n\n")
            f.write(markdown_report)


if __name__ == "__main__":
    main()