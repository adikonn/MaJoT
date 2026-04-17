import torch
import numpy as np
import sys
import os

try:
    from model import TriangularizerModel
except ImportError:
    print("❌ ОШИБКА: Не найден файл model.py или класс TriangularizerModel")
    sys.exit(1)


def calc_lower_diag_metric(M: torch.Tensor) -> float:
    """Считает среднюю сумму абсолютных значений под главной диагональю для батча."""
    lower_triangular = torch.tril(M, diagonal=-1)
    return float(torch.abs(lower_triangular).sum(dim=(1, 2)).mean().item())


def main():
    dataset_path = "dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"❌ ОШИБКА: Тестовый датасет {dataset_path} не найден!")
        sys.exit(1)

    data = np.load(dataset_path)
    A_data, B_data = data['A'], data['B']
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

        T_pred = model(A_input, B_input)

        if T_pred.shape != A_input.shape:
            print(f"❌ ОШИБКА: Модель вернула T с размерностью {T_pred.shape}, "
                  f"а ожидалось {A_input.shape}")
            sys.exit(1)

        if T_pred.dtype != torch.float64:
            print(f"❌ ОШИБКА : Модель вернула T с типом {T_pred.dtype}, "
                  f"а ожидался строго torch.float64 (Double). "
                  f"Проверьте, что в forward() вы не создаете тензоры через torch.tensor() без указания dtype.")
            sys.exit(1)
        try:
            T_inv = torch.inverse(T_pred)
        except RuntimeError:
            print("❌ ОШИБКА: Нейросеть выдала вырожденную матрицу T (Singular Matrix). "
                  "Ее невозможно обратить!")
            sys.exit(1)

        A_trans = T_inv @ A_input @ T_pred
        B_trans = T_inv @ B_input @ T_pred

        score = calc_lower_diag_metric(A_trans) + calc_lower_diag_metric(B_trans)

    print(f"\nTEST_SCORE={score:.6f}\n")

    markdown_report = (
        "| Метрика | Значение |\n"
        "| --- | --- |\n"
        f"| Количество пар (Batch) | `{n_samples}` |\n"
        f"| **Итоговая ошибка (Score)** | **`{score:.6f}`** |\n\n"
    )

    with open("metrics.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)

    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("### 🧪 Результаты тестирования (Double Precision)\n\n")
            f.write(markdown_report)


if __name__ == "__main__":
    main()