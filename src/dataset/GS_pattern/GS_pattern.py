from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ...misc.step_tracker import StepTracker
from ..types import Stage

T = TypeVar("T")

class GaussianPattern(ABC, Generic[T]):
    cfg: T
    stage: Stage
    is_overfitting: bool
    step_tracker: StepTracker | None

    def __init__(
        self,
        cfg: T,
        stage: Stage,
        is_overfitting: bool,
        step_tracker: StepTracker | None
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.is_overfitting = is_overfitting
        self.prune_ratio_value = 0.0
        self.total_nodes_value = 0
        self.step_tracker = step_tracker

    @abstractmethod
    def generate(
        self,
        context: Float[Tensor, "view 3 height width"], 
        random_threshold: float,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> dict[str, Tensor]:
        """
        Generate Gaussian Splatting parameters for each view.

        Args:
            context: Batch of images.
            device: Calculation device.

        Returns:
            A tuple of dictionaries.
            Length of tuple == context.shape[0] (number of views).
            Each dictionary contains GS parameters keys (e.g., 'means', 'scales', 'quats') 
            and their corresponding Tensor values for that specific view.
        """
        pass

    @property
    @abstractmethod
    def prune_ratio(self) -> float:
        pass
    
    @property
    @abstractmethod
    def total_nodes(self) -> int:
        pass
    
    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()