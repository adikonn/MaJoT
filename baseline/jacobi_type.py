# jacobi_type.py
# Jacobi-подобная совместная «триангуляция» двух матриц плоскими вращениями.

from __future__ import annotations

import math

import torch


def _lower_tri_residual_squared(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Квадрат нормы Фробениуса строго нижнего треугольника для A и B (вместе).

    Берём только элементы с индексами (p, q), где p > q — то, что хотим
    уменьшить, чтобы матрицы стали ближе к верхнетреугольным.
    """
    n = A.shape[0]
    # Маска: True под главной диагональю, без самой диагонали
    mask = torch.tril(torch.ones(n, n, device=A.device, dtype=torch.bool), diagonal=-1)
    # Сумма квадратов по обеим матрицам — скалярный «остаток» за один проход
    return (A[mask] ** 2).sum() + (B[mask] ** 2).sum()


def _similarity_givens_inplace(M: torch.Tensor, j: int, i: int, c: torch.Tensor, s: torch.Tensor) -> None:
    """
    Сходство: M <- G^T M G.

    G — вращение в плоскости индексов (j, i), при этом j < i.
    Встроенный 2×2 блок: G2 = [[c, -s], [s, c]] (ортогональная матрица с det=1).

    Сначала умножаем слева на G^T (линейные комбинации строк j и i),
    затем справа на G (линейные комбинации столбцов j и i).
    Остальные строки/столбцы не трогаем явно — они меняются только в позициях j, i.
    """
    # --- Шаг 1: M <- G^T M (только строки j и i) ---
    rj = M[j, :].clone()
    ri = M[i, :].clone()
    M[j, :] = c * rj + s * ri
    M[i, :] = -s * rj + c * ri

    # --- Шаг 2: M <- M G (только столбцы j и i), уже на обновлённой матрице ---
    cj = M[:, j].clone()
    ci = M[:, i].clone()
    M[:, j] = c * cj + s * ci
    M[:, i] = -s * cj + c * ci


def _right_multiply_givens_inplace(Q: torch.Tensor, j: int, i: int, c: torch.Tensor, s: torch.Tensor) -> None:
    """
    Накопление ортогонального преобразования: Q <- Q G.

    Тот же G, что в _similarity_givens_inplace: правое умножение смешивает
    только столбцы j и i матрицы Q.
    """
    qj = Q[:, j].clone()
    qi = Q[:, i].clone()
    Q[:, j] = c * qj + s * qi
    Q[:, i] = -s * qj + c * qi


def _best_givens_for_pair_min_full_residual(
    A: torch.Tensor,
    B: torch.Tensor,
    j: int,
    i: int,
    n_angles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Подобрать cos, sin для пары (j, i) по полному нижнетреугольному residual.

    Для каждого угла из сетки (включая 0 → тождественное вращение) временно
    восстанавливаем состояние пары матриц на начало шага, применяем сходство
    и считаем сумму квадратов строго поддиагональных элементов A и B.
    Выбираем угол с минимальным residual; за счёт θ=0 шаг **никогда не ухудшает**
    остаток относительно состояния до этой пары (в пределах сетки и округления).

    Перед возвратом A, B возвращаются к снимку на начало вызова — применение
    лучшего вращения делает вызывающий код один раз.
    """
    device, dtype = A.device, A.dtype

    # Снимок: до перебора углов матрицы не должны остаться «испорченными»
    A_snap = A.clone()
    B_snap = B.clone()

    # Сетка углов; первая точка — 0 (G = I); последнюю 2π не берём
    thetas = torch.linspace(
        0.0,
        2.0 * math.pi,
        n_angles + 1,
        device=device,
        dtype=dtype,
    )[:-1]
    c_all = torch.cos(thetas)
    s_all = torch.sin(thetas)

    best_res = torch.tensor(float("inf"), device=device, dtype=dtype)
    best_c = torch.ones((), device=device, dtype=dtype)
    best_s = torch.zeros((), device=device, dtype=dtype)

    for k in range(n_angles):
        # Восстанавливаем пару матриц в состояние «до этого вращения»
        A.copy_(A_snap)
        B.copy_(B_snap)
        c_k = c_all[k]
        s_k = s_all[k]
        _similarity_givens_inplace(A, j, i, c_k, s_k)
        _similarity_givens_inplace(B, j, i, c_k, s_k)
        res = _lower_tri_residual_squared(A, B)
        if res < best_res:
            best_res = res
            best_c = c_k.clone()
            best_s = s_k.clone()

    # Вернуть матрицы в состояние до перебора; реальное применение — у вызывающего
    A.copy_(A_snap)
    B.copy_(B_snap)

    return best_c, best_s


def joint_triangularize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Бейзлайн 3: Jacobi-подобный итеративный метод.
    Args:
        A: torch.Tensor, квадратная матрица размера (n, n)
        B: torch.Tensor, квадратная матрица размера (n, n)

    Returns:
        Q: torch.Tensor, ортогональная матрица размера (n, n)

    Идея: Q^(0)=I, далее на каждой паре (i,j), i>j, одно и то же вращение G
    применяется сходственно к A и B, а Q умножается справа: Q <- Q G.
    В итоге A_curr = Q^T A Q, B_curr = Q^T B Q (если стартовать с копий A, B).

    Подбор угла для пары матриц делается по **полному** нижнетреугольному residual,
    а не по одному элементу (i, j), чтобы один шаг не раздувал общий остаток.
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]

    # torch.finfo и тригонометрия — только для float; комплексные и целые не поддерживаем
    if not A.dtype.is_floating_point or not B.dtype.is_floating_point:
        raise TypeError(
            "joint_triangularize: ожидаются матрицы с dtype с плавающей точкой "
            "(float16 / bfloat16 / float32 / float64), получено "
            f"A.dtype={A.dtype}, B.dtype={B.dtype}."
        )

    n = A.shape[0]
    device = A.device
    # Общий dtype для вычислений (например float32 + float64 → float64)
    work_dtype = torch.promote_types(A.dtype, B.dtype)

    A_curr = A.to(dtype=work_dtype, device=device).clone()
    B_curr = B.to(dtype=work_dtype, device=device).clone()
    Q = torch.eye(n, dtype=work_dtype, device=device)

    # Для n <= 1 пар (i, j) с i > j нет — возвращаем I (дополнительных вращений не нужно)
    if n <= 1:
        return Q

    # Параметры остановки: лимит проходов, сетка углов, терпение к «застою»
    max_sweeps = 200
    n_angle_grid = 72
    stagnation_patience = 3
    finfo = torch.finfo(work_dtype)
    # Абсолютный порог на квадрат residual (учёт размера n и точности dtype)
    atol_sq = max(finfo.resolution * n * n, 1e-30)
    # Относительный порог: считаем, что улучшение за проход слишком маленькое
    rtol_improve = max(10.0 * finfo.eps, 1e-12)

    prev_res = _lower_tri_residual_squared(A_curr, B_curr)
    no_improve = 0

    # Внешний цикл: полные «проходы» (sweeps) по всем парам поддиагонали
    for _ in range(max_sweeps):
        # Внутренний двойной цикл: все пары (i, j) с i > j (строго нижний треугольник по позициям)
        for i in range(1, n):
            for j in range(i):
                # Угол минимизирует полный нижнетреугольный residual после пары (включая «не крутить»)
                c, s = _best_givens_for_pair_min_full_residual(
                    A_curr, B_curr, j, i, n_angle_grid
                )
                # Одно и то же сходство к обеим матрицам
                _similarity_givens_inplace(A_curr, j, i, c, s)
                _similarity_givens_inplace(B_curr, j, i, c, s)
                # Накопление произведения вращений справа
                _right_multiply_givens_inplace(Q, j, i, c, s)

        # После полного прохода оцениваем, насколько поддиагональ ещё «толстая»
        res = _lower_tri_residual_squared(A_curr, B_curr)
        if res <= atol_sq:
            # Достигли практически нулевого нижнетреугольного остатка
            break

        improve = prev_res - res
        if improve <= rtol_improve * prev_res + atol_sq:
            # Улучшение за проход слишком маленькое — возможный застой или предел точности
            no_improve += 1
            if no_improve >= stagnation_patience:
                break
        else:
            # Заметный прогресс — сбрасываем счётчик застоя
            no_improve = 0
        prev_res = res

    return Q
