# MaJoT — как использовать нейросеть (Matrix Joint Triangulation)

Этот документ описывает **как пользоваться нейросетью** MaJoT: какие тензоры ей подавать, что она возвращает, и как применить её выход для совместной триангуляции (приведения двух матриц к верхнетреугольному виду).

Репозиторий: https://github.com/adikonn/MaJoT

---

## 1) Что делает модель

Модель (класс **`TriangularizerModel`**) получает на вход пару квадратных матриц:

- `A` — `torch.Tensor` формы `(batch, n, n)`
- `B` — `torch.Tensor` формы `(batch, n, n)`

и предсказывает преобразование:

- `T` — `torch.Tensor` формы `(batch, n, n)`

после применения которого матрицы становятся **максимально близкими к верхнетреугольным** (идеально — верхнетреугольные):

- `A' = T^{-1} @ A @ T`
- `B' = T^{-1} @ B @ T`

> Важно: в текущем коде проекта используется именно `T^{-1} @ A @ T` (а не `T^TAT`). Поэтому ниже описан “практический” контракт как в репозитории.

---

## 2) Установка (минимально для инференса)

```bash
git clone https://github.com/adikonn/MaJoT.git
cd MaJoT

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# или:
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
pip install torch
```

Если будете пользоваться `evaluate.py`, может понадобиться:
```bash
pip install tqdm
```

---

## 3) Быстрое использование модели (инференс)

Ниже — базовый пример “как пользоваться нейронкой” из кода: создать модель, (опционально) загрузить веса, подать `A` и `B`, получить `T`, применить преобразование.

```python
import torch

from model.model import TriangularizerModel  # как импортируется в evaluate.py

# --- входные данные ---
# A и B: (batch, n, n)
batch, n = 4, 8
A = torch.randn(batch, n, n, dtype=torch.float64)
B = torch.randn(batch, n, n, dtype=torch.float64)

# --- модель ---
model = TriangularizerModel(n=n)

# В проекте предполагается строгий float64 (Double)
model = model.to(torch.float64)
model.eval()

# (опционально) загрузка весов
weights_path = "model_weights.pt"
# model.load_state_dict(torch.load(weights_path, map_location="cpu"))

with torch.no_grad():
    T = model(A, B)  # ожидается (batch, n, n), dtype float64

    # Проверки по контракту, которые фактически есть в evaluate.py
    assert T.shape == A.shape
    assert T.dtype == torch.float64

    # Применение преобразования
    T_inv = torch.inverse(T)
    A_tri = T_inv @ A @ T
    B_tri = T_inv @ B @ T
```

Результаты:
- `A_tri`, `B_tri` — преобразованные матрицы, которые должны быть “более верхнетреугольными”, чем исходные.

---

## 4) Контракт по входам/выходам (torch.Tensor)

### 4.1. Входы
- **`A`**: `torch.Tensor`, shape `(batch, n, n)`, dtype `torch.float64`
- **`B`**: `torch.Tensor`, shape `(batch, n, n)`, dtype `torch.float64`

Рекомендации:
- Держите `A` и `B` на одном `device` (CPU/GPU) и с одинаковым dtype.
- Если используете GPU:
  ```python
  device = torch.device("cuda")
  A = A.to(device)
  B = B.to(device)
  model = model.to(device)
  ```

### 4.2. Выход
- **`T`**: `torch.Tensor`, shape `(batch, n, n)`, dtype `torch.float64`

Ограничение из практики:
- `T` должна быть **обратимой** (иначе `torch.inverse(T)` упадёт). В `evaluate.py` это явно проверяется.

---

## 5) Как понять, что модель “сработала”

В проекте используется простая метрика “насколько матрица верхнетреугольная”: сумма модулей элементов **ниже главной диагонали**.

Можно посчитать так:

```python
def lower_diag_score(M: torch.Tensor) -> torch.Tensor:
    """
    Возвращает 1D тензор (batch,) — чем меньше, тем лучше (ближе к upper-triangular).
    """
    lower = torch.tril(M, diagonal=-1)
    return lower.abs().sum(dim=(1, 2))

score_A = lower_diag_score(A_tri)
score_B = lower_diag_score(B_tri)
score_total = score_A + score_B
print(score_total)
```

---

## 6) Использование данных из датасета (если нужно)

Если вы хотите подать в модель реальные данные как в репозитории, то `dataset/dataset.npz` содержит:
- `A`: shape `(n_samples, n, n)`
- `B`: shape `(n_samples, n, n)`
- `y`: метки (в коде используется `y == 1` для подмножества)

Пример:
```python
import numpy as np
import torch

data = np.load("dataset/dataset.npz")
A = torch.tensor(data["A"], dtype=torch.float64)
B = torch.tensor(data["B"], dtype=torch.float64)
```

Дальше можно брать батчи и подавать в модель.

---

## 7) Частые проблемы (именно при использовании модели)

### 7.1. dtype “съехал” в float32
В репозитории ожидается **строго** `torch.float64`. Если внутри `forward()` случайно создаются тензоры без `dtype=torch.float64`, результат может стать `float32`.

Решение:
- держите всё в `float64`
- если создаёте новые тензоры внутри forward — задавайте dtype/device от входов:
  ```python
  dtype = A.dtype
  device = A.device
  x = torch.zeros(..., dtype=dtype, device=device)
  ```

### 7.2. `torch.inverse(T)` падает (Singular Matrix)
Значит `T` необратима.

Что можно сделать на практике:
- добавить регуляризацию/штраф за плохую обусловленность `T`
- параметризовать `T` так, чтобы она была гарантированно обратимой (например, через произведение матриц, или через LU-подобную параметризацию)
- следить за стабильностью (double precision уже помогает)

---

## 8) Мини-резюме

- Подайте в `TriangularizerModel` тензоры `A` и `B` формы `(batch, n, n)` в `torch.float64`
- Получите `T` той же формы/типа
- Примените `T_inv @ A @ T` и `T_inv @ B @ T`
- Проверяйте качество через сумму элементов под диагональю
