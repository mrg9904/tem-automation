from __future__ import annotations

from typing import Any, TypedDict, Protocol
from contextlib import AbstractContextManager
from adapters.coordinates import ImageStageTransform
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

    def center_fov_on_image_offset(self, x_m: float, y_m: float, *,
                                   reference_stage_position_m: tuple[float, float] | None = None) -> None:
        """Center the FoV on an offset in the current image axes."""
        ...


class ScanProfileSnapshot(TypedDict):
    profile_index: int
    parameters: dict[str, Any]


@runtime_checkable
class WorkflowMicroscope(Protocol):
    """Explicit device contract for capture workflows; no silent capability fallbacks."""
    def get_stage_position(self) -> tuple[float,float]: ...
    def set_stage_position(self, x_m: float, y_m: float) -> None: ...
    def get_fov(self) -> float: ...
    def set_fov(self, fov_m: float) -> None: ...
    def get_defocus(self) -> float: ...
    def set_defocus(self, defocus_m: float) -> None: ...
    def acquire_haadf(self) -> npt.NDArray[np.float32]: ...
    def center_fov_on_image_offset(self, x_m: float, y_m: float, *, reference_stage_position_m=None) -> None: ...
    def acquisition_settings(self, *, image_size_px: int, dwell_time_us: float=1.0) -> AbstractContextManager[None]: ...
    def get_coordinate_transform(self) -> ImageStageTransform: ...
    def get_last_scan_profile(self) -> ScanProfileSnapshot | None: ...
    def persist_last_scan_profile(self, snapshot: ScanProfileSnapshot | None=None) -> ScanProfileSnapshot | None: ...


def require_workflow_microscope(microscope):
    methods = ('get_stage_position','set_stage_position','get_fov','set_fov',
        'get_defocus','set_defocus','acquire_haadf','center_fov_on_image_offset',
        'acquisition_settings','get_coordinate_transform','get_last_scan_profile',
        'persist_last_scan_profile')
    missing=[name for name in methods if not callable(getattr(microscope,name,None))]
    if missing:
        raise TypeError('Workflow microscope missing capabilities: '+', '.join(missing))
