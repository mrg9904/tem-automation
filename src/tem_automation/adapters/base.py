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

    def acquire_haadf(self) -> npt.NDArray[np.float32]:
        """Acquire and return one 2-D HAADF image."""
        ...
