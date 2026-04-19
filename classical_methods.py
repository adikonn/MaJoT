from __future__ import annotations

from itertools import product
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _all_words(max_len: int) -> List[Tuple[int, ...]]:
    """Generate all non-empty words over alphabet {0, 1} up to max_len."""
    words: List[Tuple[int, ...]] = []
    for length in range(1, max_len + 1):
        words.extend(product((0, 1), repeat=length))
    return words


def _word_to_matrix(word: Sequence[int], A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Convert a binary word to matrix product where 0->A, 1->B."""
    result = np.eye(A.shape[0], dtype=np.complex128)
    for token in word:
        result = result @ (A if token == 0 else B)
    return result


def _commutators_from_words(words: Iterable[Tuple[int, ...]], A: np.ndarray, B: np.ndarray) -> List[np.ndarray]:
    """Build commutators [W_i, W_j] for all i < j."""
    matrices = [_word_to_matrix(w, A, B) for w in words]
    commutators: List[np.ndarray] = []
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            commutators.append(matrices[i] @ matrices[j] - matrices[j] @ matrices[i])
    return commutators


def _stack_rank(mats: Sequence[np.ndarray], n: int) -> int:
    if not mats:
        return 0
    stacked = np.stack([m.reshape(n * n) for m in mats], axis=0)
    return int(np.linalg.matrix_rank(stacked, tol=1e-10))


def check_pair_triangularizable(
    A: np.ndarray,
    B: np.ndarray,
    *,
    tol_trace: float = 1e-8,
    return_details: bool = False,
) -> bool | Tuple[bool, dict]:
    """
    Heuristic/classical-style check for simultaneous triangularizability.

    We implement a practical variant inspired by the trace-commutator criterion:
    - build binary words up to length n over {A, B};
    - compute commutators [W_i, W_j] and their traces;
    - compute growth of span dim(B(n)) - dim(B(n-1)) using rank of flattened words.
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape")

    n = A.shape[0]
    Ac = np.asarray(A, dtype=np.complex128)
    Bc = np.asarray(B, dtype=np.complex128)

    words_n = _all_words(n)
    words_n_1 = _all_words(max(n - 1, 1))

    commutators = _commutators_from_words(words_n, Ac, Bc)
    max_abs_trace = max((abs(np.trace(c)) for c in commutators), default=0.0)
    trace_pass = max_abs_trace < tol_trace

    mats_n = [_word_to_matrix(w, Ac, Bc) for w in words_n]
    mats_n_1 = [_word_to_matrix(w, Ac, Bc) for w in words_n_1]
    growth = _stack_rank(mats_n, n) - _stack_rank(mats_n_1, n)
    growth_pass = growth == 0

    is_triangularizable = bool(trace_pass and growth_pass)

    if return_details:
        return is_triangularizable, {
            "max_abs_trace_commutator": float(max_abs_trace),
            "dim_growth_Bn_minus_Bn_1": int(growth),
            "trace_pass": bool(trace_pass),
            "growth_pass": bool(growth_pass),
        }
    return is_triangularizable

