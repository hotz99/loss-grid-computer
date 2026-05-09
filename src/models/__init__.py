from src.system_schema import MLTaskSpec
from src.models.mlp_regressor import build_model as build_mlp_regressor_model
from src.models.resnet20 import build_model as build_resnet20_model


def build_model(spec: MLTaskSpec):
    if spec.model == "resnet20":
        return build_resnet20_model(spec)
    if spec.model == "mlp_regressor":
        return build_mlp_regressor_model(spec)
    raise NotImplementedError(
        f"Model builder not implemented for workload {spec.name}"
    )

__all__ = [
    "build_model",
]
