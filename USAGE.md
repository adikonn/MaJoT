# MaJoT — установка, запуск и формат данных

Этот файл дополняет основной `README.md` и описывает практические шаги: как установить зависимости, как запустить оценку модели и в каком формате проект ожидает данные.

Репозиторий: https://github.com/adikonn/MaJoT

---

## 1) Требования

- Python **3.9+**
- `numpy`
- `torch` (PyTorch)
- (опционально) `tqdm` — используется в `evaluate.py` для прогресс-бара

В репозитории есть файл `requirements.txt`, но **torch туда намеренно не добавлен** (см. комментарий в файле).

---

## 2) Установка

### 2.1. Клонирование
```bash
git clone https://github.com/adikonn/MaJoT.git
cd MaJoT
```

### 2.2. Виртуальное окружение (рекомендуется)
**Linux/macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.3. Зависимости Python
Установите зависимости из `requirements.txt`:
```bash
pip install -r requirements.txt
```

Установите PyTorch отдельно (CPU или CUDA — по вашей среде). Пример для CPU:
```bash
pip install torch
```

Если при запуске `evaluate.py` будет ошибка `ModuleNotFoundError: No module named 'tqdm'`, установите:
```bash
pip install tqdm
```

---

## 3) Как пользоваться (что можно запустить)

В репозитории есть скрипт **`evaluate.py`**, который:
- загружает тестовый датасет `dataset/dataset.npz`
- создаёт модель `TriangularizerModel`
- (если найден) загружает веса из `model_weights.pt`
- прогоняет батчами и считает метрику “насколько матрицы стали верхнетреугольными”
- сохраняет отчёт в `metrics.md`
- (в GitHub Actions) пишет summary через `GITHUB_STEP_SUMMARY`

### 3.1. Запуск оценки
Из корня репозитория:
```bash
python evaluate.py
```

Ожидаемые файлы/пути (как в коде `evaluate.py`):
- `dataset/dataset.npz` — **обязателен**
- `model_weights.pt` — **опционален** (если нет, будет предупреждение и тестирование случайных весов)

---

## 4) Формат входных данных (torch.Tensor) — как устроено в этом проекте

Согласно `evaluate.py`, входом для модели выступают **две квадратные матрицы** `A` и `B`, подаваемые батчами.

### 4.1. Файл датасета: `dataset/dataset.npz`

`evaluate.py` ожидает, что `np.load("dataset/dataset.npz")` содержит ключи:

- `A`: numpy-массив формы **`(n_samples, n, n)`**
- `B`: numpy-массив формы **`(n_samples, n, n)`**
- `y`: numpy-массив меток (в коде используется `y == 1` как маска “совместно триангуляризуемых”)

Пример чтения (упрощённо):
```python
import numpy as np

data = np.load("dataset/dataset.npz")
A_data, B_data = data["A"], data["B"]
y_data = data["y"]
```

### 4.2. Тензоры, которые подаются в модель

Внутри `evaluate.py` numpy-массивы конвертируются в PyTorch тензоры **строго double precision**:

- `A_input = torch.tensor(A_data, dtype=torch.float64)`
- `B_input = torch.tensor(B_data, dtype=torch.float64)`

То есть контракт такой:

- **Тип:** `torch.Tensor`
- **dtype:** `torch.float64` (это важно — в коде есть явная проверка)
- **shape:** `(batch, n, n)` для батча, и `(n_samples, n, n)` для всего набора

### 4.3. Что должна вернуть модель

`TriangularizerModel(A_batch, B_batch)` должен вернуть тензор `T_batch`:

- **shape:** должен совпадать с `A_batch.shape` (т.е. `(batch, n, n)`)
- **dtype:** должен быть `torch.float64`

Дальше в `evaluate.py` считается:
- `T_inv = torch.inverse(T_batch)`
- `A_trans = T_inv @ A_batch @ T_batch`
- `B_trans = T_inv @ B_batch @ T_batch`

И метрика — сумма модулей элементов **ниже главной диагонали** (чем меньше, тем “более верхнетреугольная” матрица):
```python
lower_triangular = torch.tril(M, diagonal=-1)
score = torch.abs(lower_triangular).sum(dim=(1, 2))  # по каждой матрице в батче
```

---

## 5) Типичные ошибки и как их чинить

### 5.1. Нет датасета
Ошибка из `evaluate.py`:
- `❌ ОШИБКА: Тестовый датасет dataset/dataset.npz не найден!`

Решение: убедитесь, что файл существует по пути `dataset/dataset.npz` и вы запускаете скрипт из корня репозитория.

### 5.2. Модель вернула не float64
В `evaluate.py` есть проверка:
- модель обязана возвращать `torch.float64`

Если вы меняете `forward()` и где-то создаёте тензоры через `torch.tensor(...)` без `dtype=torch.float64`, тип может “съехать” в `float32`.

Решение: следить за dtype и устройством, использовать `.to(torch.float64)` или явно задавать `dtype=torch.float64` при создании тензоров.

### 5.3. Вырожденная матрица `T` (Singular Matrix)
Если `torch.inverse(T_batch)` падает — значит `T` необратима.

Решение: регуляризация/ограничения на модель, контроль кондиционирования, штрафы в loss, и т.д. (зависит от обучения).

---

## 6) Что генерирует скрипт

После успешного прогона создаётся файл:

- `metrics.md` — таблица с метриками (General Score, Triang Score и т.п.)

В консоль также печатается:
- `GENERAL_SCORE=...`
- `TRIANG_SCORE=...`