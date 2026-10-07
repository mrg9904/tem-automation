from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Microscope(Protocol):
    """Small, vendor-independent API needed by the autofocus algorithm."""

    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def get_defocus(self) -> float:
        """Return defocus in meters."""
        ...

    def set_defocus(self, defocus_m: float) -> None:
        """Set defocus in meters."""
        ...

    def get_fov(self) -> float:
        """Return the active acquisition field of view in meters."""
        ...

    def acquire_haadf(self) -> npt.NDArray[np.float32]:
        """Acquire and return one 2-D HAADF image."""
        ...


class FoVPositioningMicroscope(Protocol):
    """Positioning API; (x, y) tuples and all distances are in meters."""

    def get_fov(self) -> float:
        ...

    def set_fov(self, fov_m: float) -> None:
        ...

    def get_stage_position(self) -> tuple[float, float]:
        ...

    def set_stage_position(self, x_m: float, y_m: float) -> None:
        ...

    def center_fov_on_image_offset(self, x_m: float, y_m: float) -> None:
        """Center the FoV on an offset in the current image axes."""
        ...
