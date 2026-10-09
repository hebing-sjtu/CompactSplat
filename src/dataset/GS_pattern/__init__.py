from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .GS_pattern import GaussianPattern
from .GS_pattern_quadtree import GaussianPatternQuadTree, GaussianPatternQuadTreeCfg
from .GS_pattern_quadtree_pro import GaussianPatternQuadTreePro, GaussianPatternQuadTreeProCfg
from .GS_pattern_rectgrowth import GaussianPatternRectGrowth, GaussianPatternRectGrowthCfg
from .GS_pattern_rectgrowth_pro import GaussianPatternRectGrowthPro, GaussianPatternRectGrowthProCfg

GS_PATTERNS: dict[str, GaussianPattern[Any]] = {
    "quadtree": GaussianPatternQuadTree,
    "quadtree_pro": GaussianPatternQuadTreePro,
    "rectgrowth": GaussianPatternRectGrowth,
    "rectgrowth_pro": GaussianPatternRectGrowthPro,
}

GaussianPatternCfg = (
    GaussianPatternQuadTreeCfg
    | GaussianPatternQuadTreeProCfg
    | GaussianPatternRectGrowthCfg
    | GaussianPatternRectGrowthProCfg
)

def get_GS_pattern(
    cfg: GaussianPatternCfg,
    stage: Stage,
    overfit: bool,
    step_tracker: StepTracker | None,
) -> GaussianPattern[Any]:
    return GS_PATTERNS[cfg.name](
        cfg,
        stage,
        overfit,
        step_tracker,
    )