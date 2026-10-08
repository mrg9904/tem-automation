"""Explicit image/stage geometry shared by movement and mosaics."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ImageStageTransform:
    rotation_rad: float = 0.0
    stage_direction: float = -1.0

    def __post_init__(self):
        if not np.isfinite(self.rotation_rad) or self.stage_direction not in (-1.,1.):
            raise ValueError('Invalid image/stage transform')

    @property
    def rotation(self):
        c,s=np.cos(self.rotation_rad),np.sin(self.rotation_rad)
        return np.array([[c,-s],[s,c]])

    @staticmethod
    def _vector(value):
        value=np.asarray(value,dtype=float)
        if value.shape!=(2,) or not np.isfinite(value).all():
            raise ValueError('Expected two finite coordinates')
        return value

    def stage_position_for_offset(self, offset_m, reference_stage_m):
        return self._vector(reference_stage_m)+self.stage_direction*self.rotation@self._vector(offset_m)

    def image_offset_for_stage_position(self, stage_m, reference_stage_m):
        delta=self._vector(stage_m)-self._vector(reference_stage_m)
        return self.stage_direction*self.rotation.T@delta
