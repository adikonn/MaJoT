<div align="center">

# 🔺 MaJoT: Matrix Joint Triangulation 🔺

**Построение нейросетевого инструмента для совместной триангуляции двух матриц**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 О проекте

В многомерном анализе данных часто требуется найти такое преобразование, которое одновременно приводит две матрицы к удобному для дальнейшей работы виду (в этом проекте - к верхнетреугольному).

### 🧮 Постановка задачи

Пусть даны две квадратные матрицы $A$ и $B$. Наша нейросеть обучается находить такую обратимую матрицу преобразования $T$, при которой новые матрицы:

$$ A' = T^TAT $$
$$ B' = T^TBT $$

становятся **одновременно верхнетреугольными** (или максимально близкими к таковым, минимизируя веса под главной диагональю).

---

## 🧱 Что реализовано

В проекте реализованы два подхода и их сравнение:

- **Нейросетевой подход**
  - Модель `TriangularizerModel` в `model/model.py` возвращает обратимую матрицу через `T = exp(M)` (матричная экспонента гарантирует обратимость).
  - Обучение в `train.py` минимизирует величину поддиагональных элементов для пар `T^TAT` и `T^TBT`.
  - Добавлен **label-aware margin loss** для лучшего разделения классов (триангулизуемые/нетриангулизуемые).
  - Есть валидация, early stopping, сохранение лучших весов и истории обучения.

- **Классический критерий**
  - В `classical_methods.py` реализована практическая проверка совместной триангулизуемости:
    - генерация слов над `{A, B}` до длины `n`,
    - проверка следов коммутаторов,
    - контроль роста `dim(B(n)) - dim(B(n-1))`.

- **Сравнение методов**
  - В `evaluate.py` добавлена оценка `NN vs Classical`.
  - Порог для NN подбирается на calibration split и затем проверяется на holdout-части.
  - Считаются `Accuracy`, `Precision`, `Recall`, `F1`, confusion matrix.

- **Авто-исследование**
  - `run_experiments.py`: мульти-сид прогоны и агрегирование метрик.
  - `auto_research.py`: ablation/sweep нескольких конфигов, ранжирование и выбор лучшего.

---

## 📂 Структура

- `dataset/generate_data.py` - генерация датасета (включая hard negatives).
- `model/model.py` - нейросетевая модель, возвращающая преобразование `T`.
- `train.py` - обучение модели.
- `evaluate.py` - оценка качества и сравнение с классикой.
- `classical_methods.py` - классический критерий совместной триангулизуемости.
- `run_experiments.py` - мульти-сид эксперименты.
- `auto_research.py` - автоматический sweep/ablation и ранжирование конфигов.

---

## 🚀 Быстрый старт

1. Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

2. Сгенерировать датасет:

```bash
python dataset/generate_data.py --n-joint 300 --n-random 300 --n-size 3 --seed 42
```

3. Обучить модель:

```bash
python train.py --epochs 120 --batch-size 64 --lr 0.0015 --patience 20 --class-margin-weight 0.35 --class-margin 0.4 --seed 42
```

4. Оценить:

```bash
python evaluate.py --batch-size 256 --calib-ratio 0.25 --seed 42
```

5. Прогнать мульти-сид benchmark:

```bash
python run_experiments.py --seeds 7 13 42 --epochs 50 --patience 12 --batch-size 64 --lr 0.0015 --class-margin-weight 0.35 --class-margin 0.4
```

6. Прогнать полный авто-sweep/ablation:

```bash
python auto_research.py --seeds 7 13 42 --n-joint 300 --n-random 300 --n-size 3
```

---

## 👶 Инструкция для полных чайников

Ниже самый простой путь, если вы вообще не программист.

### 1) Один раз установить Python

- Скачайте Python с официального сайта: [python.org](https://www.python.org/downloads/).
- При установке обязательно включите галочку **Add Python to PATH**.

### 2) Открыть папку проекта

- Распакуйте/склонируйте проект в любую папку (например `D:\proj\MaJoT`).
- Откройте эту папку в терминале (PowerShell/Command Prompt).

### 3) Установить все нужные библиотеки

```bash
python -m pip install -r requirements.txt
```

### 4) Проверить, что есть файл весов

- В корне проекта должен быть файл `model_weights.pt`.
- Если его нет, сначала обучите модель командой:

```bash
python train.py
```

### 5) Запустить простое окно (GUI)

```bash
python app_gui.py
```

Откроется окно, где:
- слева вставляете матрицу `A`,
- справа вставляете матрицу `B`,
- нажимаете **"Применить нейронку"**,
- внизу получаете:
  - матрицу `T`,
  - преобразованные `A'` и `B'`,
  - числовую оценку качества (score).

### Как вводить матрицы

Можно так:

```text
1 2 3
0 4 5
0 0 6
```

Или так:

```text
1,2,3;0,4,5;0,0,6
```

Главное:
- матрица должна быть квадратной;
- `A` и `B` должны быть одинакового размера.

---

## 🖥️ Как сделать .exe (Windows)

Для сборки exe есть готовый файл `build_exe.bat`.

Просто выполните:

```bat
build_exe.bat
```

После сборки получите:

- `dist\MaJoT-GUI.exe`

Это обычное desktop-приложение: можно запускать двойным кликом.

---

## 📊 Результаты проделанной работы

По авто-исследованию (см. `auto_research_report.md` / `auto_research_results.json`) лучшим оказался конфиг:

- `hard_neg_margin_strong`
  - `nn_acc_mean = 0.523704`
  - `nn_f1_mean = 0.660340`
  - `classical_acc_mean = 0.517037`
  - `classical_f1_mean = 0.516680`

Итог: в текущей экспериментальной постановке нейросетевой подход стабильно опережает классический по классификационным метрикам.

---

## 🧪 Полезные артефакты

- `model_weights.pt` - актуальные веса модели.
- `train_history.json` - история обучения последнего запуска.
- `metrics.md` - отчет последней оценки.
- `auto_research_report.md` - итоговый авто-отчет.
- `auto_research_results.json` - подробные результаты sweep/ablation.
- `app_gui.py` - простое desktop GUI для ручного ввода матриц.
- `build_exe.bat` - скрипт сборки `MaJoT-GUI.exe`.

---

## 🏆 Команда проекта: «ЧЕМПИОНЫ»


| Имя | Роль в проекте | GitHub / Контакты |
| :--- | :--- | :--- |
| 🧑‍💻 **Калмурзин Адилет** | *MLOps & ML Engineer* | [@adikonn](#) |
| 🧮 **Никитин Владимир** | *роль* | [@metroooooosumma](#) |
| ⚙️ **Подвысоцкий Константин**| *роль* | [@Samibaduk](#) |
| 📈 **Тепляков Роман** | *VIBE ML CODING* | [@Romandzella](#) |

<a href="https://github.com/adikonn/MaJoT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=adikonn/MaJoT" />
</a>

---
<div align="center">
  <i>Сделано с ❤️ командой ЧЕМПИОНЫ 🏆🏆🏆</i>
</div>
