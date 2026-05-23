"""Реестр моделей. Новые архитектуры регистрируются здесь для train.py и бенчмарков."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import Triangularizer
from .dual_stream_rowcol import DualStreamRowCol
from .matrix_transformer import MatrixTransformer

_REGISTRY: dict[str, type[nn.Module]] = {
    "matrix_transformer": MatrixTransformer,
    "dual_stream_rowcol": DualStreamRowCol,
}


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Собрать модель по конфигу: поле `name` — ключ реестра, остальное — kwargs конструктора."""
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name](**cfg)


__all__ = [
    "build_model",
    "Triangularizer",
    "MatrixTransformer",
    "DualStreamRowCol",
]
