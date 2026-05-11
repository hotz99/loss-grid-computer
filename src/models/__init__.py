from src.schemas import MLTaskSpec
from src.models.mnist_mlp import build_model as build_mnist_mlp_model
from src.models.mlp_regressor import build_model as build_mlp_regressor_model
from src.models.resnet20 import build_model as build_resnet20_model
from src.models.row_gru import build_model as build_row_gru_model


def build_model(spec: MLTaskSpec):
    if spec.model == "resnet20":
        return build_resnet20_model(spec)
    if spec.model == "mlp_regressor":
        return build_mlp_regressor_model(spec)
    if spec.model == "row_gru":
        return build_row_gru_model(spec)
    if spec.model == "mnist_mlp":
        return build_mnist_mlp_model(spec)
    raise NotImplementedError(
        f"Model builder not implemented for workload {spec.name}"
    )

__all__ = [
    "build_model",
]
