from src.models.resnet20 import build_model
from src.models.resnet20_compiled import (
    build_resnet20_compiled_chunk_evaluator,
)

__all__ = [
    "build_model",
    "build_resnet20_compiled_chunk_evaluator",
]
