import os
import sys

import numpy as np
import torch
from tqdm import tqdm

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


def main():
    dataset_path = "dataset/dataset.npz"
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

    weights_path = "model_weights.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"✅ Веса модели ({weights_path}) успешно загружены.")
    else:
        print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Файл {weights_path} не найден. Тестируем случайные веса!")

    model.eval()

    with torch.no_grad():
        A_input = torch.tensor(A_data, dtype=torch.float64)
        B_input = torch.tensor(B_data, dtype=torch.float64)

        batch_size = 256  # Можно изменить размер батча под вашу память
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
            try:
                T_inv = torch.inverse(T_batch)
            except RuntimeError:
                print("\n❌ ОШИБКА: Нейросеть выдала вырожденную матрицу T (Singular Matrix). "
                      "Ее невозможно обратить!")
                sys.exit(1)

            A_trans = T_inv @ A_batch @ T_batch
            B_trans = T_inv @ B_batch @ T_batch

            scores_A = calc_lower_diag_metric_batch(A_trans)
            scores_B = calc_lower_diag_metric_batch(B_trans)
            total_scores_list.append(scores_A + scores_B)

        total_scores = torch.cat(total_scores_list, dim=0)

    y_tensor = torch.tensor(y_data, dtype=torch.int32)

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
    print("=" * 40 + "\n")

    markdown_report = (
        "| Метрика | Значение |\n"
        "| --- | --- |\n"
        f"| Всего пар в датасете | `{n_samples}` |\n"
        f"| Из них строго триангуляризуемых (y=0) | `{triang_count}` |\n"
        f"| **Итоговая ошибка (General Score)** | **`{avg_total_score:.6f}`** |\n"
        f"| **Ошибка на трианг. матрицах (Triang Score)** | **`{avg_triang_score:.6f}`** |\n"
        f"| **Мин. ошибка на трианг. матрицах (Triang Score)** | **`{min_triang_score:.6f}`** |\n"
        f"| **Макс. ошибка на трианг. матрицах (Triang Score)** | **`{max_triang_score:.6f}`** |\n\n"
    )

    # Сохраняем в файл metrics.md для бота
    with open("metrics.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Дублируем в Summary GitHub Actions
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("### 🧪 Результаты тестирования (Double Precision)\n\n")
            f.write(markdown_report)


if __name__ == "__main__":
    main()