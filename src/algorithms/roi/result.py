from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FindROIResult:
    image: np.ndarray
    fov_m: float
    score_map: np.ndarray
    labels: np.ndarray
    candidates: np.ndarray
    threshold: float
    model_id: str
    descriptions: tuple[str, ...] = ()
    reference_score_map: np.ndarray | None = None
    context_baseline_map: np.ndarray | None = None
    threshold_map: np.ndarray | None = None

    def save(self, directory):
        from .output import save_roi_result
        return save_roi_result(self, directory)
