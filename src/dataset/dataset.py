from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg
from .GS_pattern import GaussianPatternCfg


@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
    GS_pattern: GaussianPatternCfg
