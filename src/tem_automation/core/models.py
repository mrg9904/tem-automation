from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Mapping

import numpy as np
import numpy.typing as npt


class ImagingMode(str, Enum):
    HAADF_STEM = "haadf_stem"


@dataclass(frozen=True)
class AcquisitionSettings:
    mode: ImagingMode = ImagingMode.HAADF_STEM
    fov_m: float = 100e-9
    width_px: int = 256
    height_px: int = 256
    dwell_time_s: float = 1e-6
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.fov_m <= 0:
            raise ValueError("fov_m must be positive")
        if self.width_px <= 1 or self.height_px <= 1:
            raise ValueError("image dimensions must be greater than one pixel")
        if self.dwell_time_s <= 0:
            raise ValueError("dwell_time_s must be positive")


@dataclass(frozen=True)
class MicroscopeState:
    defocus_m: float


@dataclass(frozen=True)
class ImageFrame:
    data: npt.NDArray[np.float32]
    pixel_size_m: tuple[float, float]
    timestamp_s: float
    metadata: Mapping[str, Any]

