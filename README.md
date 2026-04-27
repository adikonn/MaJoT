
<div align="center">

# 🔺 MaJoT: Matrix Joint Triangulation 🔺

**Построение нейросетевого инструмента для совместной триангуляции двух матриц**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 О проекте

В многомерном анализе данных часто требуется найти такое преобразование, которое одновременно приводит две матрицы к «удобному» для дальнейшей работы и вычислений виду (в данном случае, к верхнетреугольному). 

### 🧮 Постановка задачи

Пусть даны две квадратные матрицы $A$ и $B$. Наша нейросеть обучается находить такую обратимую матрицу преобразования $T$, при которой новые матрицы:

$$ A' = T^TAT $$
$$ B' = T^TBT $$

становятся **одновременно верхнетреугольными** (или максимально близкими к таковым, минимизируя веса под главной диагональю).

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

---

## 🚀 Быстрый запуск (большой датасет + глубокая архитектура)

### 1) Генерация большого датасета

```bash
python dataset/generate_data.py --n-size 3 --n-joint 50000 --n-non-joint 50000 --seed 42 --output dataset/dataset.npz
```

### 2) Обучение модели

```bash
python model/train.py --dataset dataset/dataset.npz --weights model_weights.pt --epochs 40 --batch-size 512 --lr 1e-3
```

### 3) Оценка

```bash
python evaluate.py
```

Новая архитектура использует более глубокую residual MLP и гарантирует обратимость матрицы преобразования $T$ через факторизацию $T = L \cdot U$ с положительной диагональю у $U$.
