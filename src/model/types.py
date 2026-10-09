from dataclasses import dataclass

from jaxtyping import Float, Bool
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales_mul: Float[Tensor, "batch gaussian 2"]
    normals: Float[Tensor, "batch gaussian 3"]
    valid_mask: Bool[Tensor, "batch gaussian"]
