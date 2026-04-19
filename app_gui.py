import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Tuple

import numpy as np
import torch

from model.model import TriangularizerModel


def parse_matrix(text: str) -> np.ndarray:
    """
    Supports formats:
    - rows separated by newline: "1 2 3\\n4 5 6\\n7 8 9"
    - rows separated by semicolon: "1,2,3;4,5,6;7,8,9"
    """
    raw = text.strip()
    if not raw:
        raise ValueError("Матрица пустая.")
    rows = [r.strip() for r in raw.replace(";", "\n").splitlines() if r.strip()]
    matrix = []
    for row in rows:
        tokens = [x for x in row.replace(",", " ").split() if x]
        matrix.append([float(t) for t in tokens])
    arr = np.array(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Матрица должна быть квадратной.")
    return arr


def lower_score(M: np.ndarray) -> float:
    return float(np.abs(np.tril(M, k=-1)).sum())


class TriangulationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MaJoT - Совместная триангуляция матриц")
        self.root.geometry("1100x760")

        self.weights_path_var = tk.StringVar(value="model_weights.pt")

        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(top, text="Файл весов модели:").pack(side=tk.LEFT)
        tk.Entry(top, textvariable=self.weights_path_var, width=70).pack(side=tk.LEFT, padx=6)
        tk.Button(top, text="Выбрать", command=self._pick_weights).pack(side=tk.LEFT)

        matrix_frame = tk.Frame(self.root)
        matrix_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=8)

        left = tk.Frame(matrix_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        tk.Label(left, text="Матрица A").pack(anchor="w")
        self.a_text = scrolledtext.ScrolledText(left, height=12)
        self.a_text.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(matrix_frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        tk.Label(right, text="Матрица B").pack(anchor="w")
        self.b_text = scrolledtext.ScrolledText(right, height=12)
        self.b_text.pack(fill=tk.BOTH, expand=True)

        hint = (
            "Формат ввода: числа через пробел или запятую, строки через Enter или ';'.\n"
            "Пример:\n"
            "1 2 3\n"
            "0 4 5\n"
            "0 0 6"
        )
        tk.Label(self.root, text=hint, justify=tk.LEFT, fg="gray").pack(anchor="w", padx=12)

        actions = tk.Frame(self.root)
        actions.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(actions, text="Применить нейронку", command=self.run_inference, height=2).pack(side=tk.LEFT)
        tk.Button(actions, text="Подставить пример", command=self._insert_example).pack(side=tk.LEFT, padx=8)
        tk.Button(actions, text="Очистить", command=self._clear).pack(side=tk.LEFT, padx=8)

        tk.Label(self.root, text="Результаты").pack(anchor="w", padx=10)
        self.out_text = scrolledtext.ScrolledText(self.root, height=20)
        self.out_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    def _pick_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл весов",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.weights_path_var.set(path)

    def _insert_example(self) -> None:
        self.a_text.delete("1.0", tk.END)
        self.b_text.delete("1.0", tk.END)
        self.a_text.insert(tk.END, "2 1 0\n0 3 4\n0 0 5")
        self.b_text.insert(tk.END, "1 0 1\n0 2 1\n0 0 4")

    def _clear(self) -> None:
        self.a_text.delete("1.0", tk.END)
        self.b_text.delete("1.0", tk.END)
        self.out_text.delete("1.0", tk.END)

    def _load_model(self, n: int, weights_path: str) -> TriangularizerModel:
        model = TriangularizerModel(n=n).to(torch.float64)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def _infer(self, A: np.ndarray, B: np.ndarray, weights_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self._load_model(A.shape[0], weights_path)
        with torch.no_grad():
            A_t = torch.tensor(A, dtype=torch.float64).unsqueeze(0)
            B_t = torch.tensor(B, dtype=torch.float64).unsqueeze(0)
            T_t = model(A_t, B_t)[0]
            T = T_t.detach().cpu().numpy()
            A_new = T.T @ A @ T
            B_new = T.T @ B @ T
        return T, A_new, B_new

    def run_inference(self) -> None:
        try:
            A = parse_matrix(self.a_text.get("1.0", tk.END))
            B = parse_matrix(self.b_text.get("1.0", tk.END))
            if A.shape != B.shape:
                raise ValueError("Матрицы A и B должны быть одинакового размера.")
            weights_path = self.weights_path_var.get().strip()
            if not weights_path:
                raise ValueError("Укажите путь к файлу весов.")

            T, A_new, B_new = self._infer(A, B, weights_path)

            s_a = lower_score(A_new)
            s_b = lower_score(B_new)
            total = s_a + s_b

            self.out_text.delete("1.0", tk.END)
            self.out_text.insert(tk.END, "=== Преобразование найдено ===\n\n")
            self.out_text.insert(tk.END, f"Размер матриц: {A.shape[0]}x{A.shape[1]}\n")
            self.out_text.insert(tk.END, f"Score(A') = {s_a:.6f}\n")
            self.out_text.insert(tk.END, f"Score(B') = {s_b:.6f}\n")
            self.out_text.insert(tk.END, f"Total score = {total:.6f}\n\n")
            self.out_text.insert(tk.END, "T =\n")
            self.out_text.insert(tk.END, np.array2string(T, precision=5, suppress_small=False))
            self.out_text.insert(tk.END, "\n\nA' = T^T A T\n")
            self.out_text.insert(tk.END, np.array2string(A_new, precision=5, suppress_small=False))
            self.out_text.insert(tk.END, "\n\nB' = T^T B T\n")
            self.out_text.insert(tk.END, np.array2string(B_new, precision=5, suppress_small=False))

        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))


def main() -> None:
    root = tk.Tk()
    app = TriangulationApp(root)
    app._insert_example()
    root.mainloop()


if __name__ == "__main__":
    main()
