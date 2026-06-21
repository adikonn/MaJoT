"""Model registry. Add new architectures here so that train.py can pick them up by name."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Triangularizer
from .cross_attn_triangularizer import CrossAttnTriangularizer
from .dual_stream_rowcol import DualStreamRowCol
from .dual_stream_rowcol_ortho import DualStreamRowColOrtho
from .equivariant_matrix_net import EquivariantMatrixNet
from .iterative_refinement import IterativeRefinementTriangularizer
from .iterative_refinement_ortho import IterativeRefinementOrtho
from .learned_givens import LearnedGivens
from .matrix_transformer import MatrixTransformer
from .matrix_transformer_ortho import MatrixTransformerOrtho

if TYPE_CHECKING:
    from torch import nn

_REGISTRY: dict[str, type[nn.Module]] = {
    "matrix_transformer": MatrixTransformer,
    "matrix_transformer_ortho": MatrixTransformerOrtho,
    "dual_stream_rowcol": DualStreamRowCol,
    "dual_stream_rowcol_ortho": DualStreamRowColOrtho,
    "iterative_refinement": IterativeRefinementTriangularizer,
    "iterative_refinement_ortho": IterativeRefinementOrtho,
    "equivariant_matrix_net": EquivariantMatrixNet,
    "cross_attn_triangularizer": CrossAttnTriangularizer,
    "learned_givens": LearnedGivens,
}


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Construct a model from its config dictionary.

    The config must contain a `name` key pointing to a registered model class; the
    remaining keys are forwarded as constructor kwargs.
    """
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in _REGISTRY:
        msg = f"Unknown model '{name}'. Registered: {list(_REGISTRY)}"
        raise KeyError(msg)
    return _REGISTRY[name](**cfg)


__all__ = [
    "CrossAttnTriangularizer",
    "DualStreamRowCol",
    "DualStreamRowColOrtho",
    "EquivariantMatrixNet",
    "IterativeRefinementOrtho",
    "IterativeRefinementTriangularizer",
    "LearnedGivens",
    "MatrixTransformer",
    "MatrixTransformerOrtho",
    "Triangularizer",
    "build_model",
]
