"""Model registry. Add new architectures here so that train.py can pick them up by name."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import Triangularizer
from .matrix_transformer import MatrixTransformer

_REGISTRY: dict[str, type[nn.Module]] = {
    "matrix_transformer": MatrixTransformer,
}


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Construct a model from its config dictionary.

    The config must contain a `name` key pointing to a registered model class; the
    remaining keys are forwarded as constructor kwargs.
    """
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name](**cfg)


__all__ = ["build_model", "Triangularizer", "MatrixTransformer"]
