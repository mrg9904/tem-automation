"""Center the current FoV on a selected image position and fit its size."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adapters.base import FoVPositioningMicroscope


@dataclass(frozen=True)
class Zoom2FitResult:
    center_offset_m: tuple[float, float]
    diameter_m: float
    requested_fov_m: float
    actual_fov_m: float
    original_stage_position_m: tuple[float, float]
    actual_stage_position_m: tuple[float, float]


def zoom_to_fit(
    microscope: FoVPositioningMicroscope,
    center_m: tuple[float, float],
    diameter_m: float,
    *,
    padding: float = 1.0,
    reference_stage_position_m: tuple[float, float] | None = None,
) -> Zoom2FitResult:
    """Center on (x, y) relative to the CURRENT image center, then fit FoV.

    All distances are meters. Directly accepts FindParticles offset_x_m,
    offset_y_m and circle_diameter_m. padding >= 1 adds optional margin.
    The adapter converts image axes and direction to specimen/stage motion.
    This action changes the active profile FoV without acquiring an image
    or changing defocus. Returned FoV is the instrument's actual readback.

    Offsets refer to the image BEFORE this action. Do not reuse an old image's
    offsets after moving the stage, changing scan center, or rotating the scan.
    reference_stage_position_m optionally anchors offsets to a saved reference
    stage position. The adapter moves directly to that absolute target without
    revisiting the reference. Scan rotation/center and beam shift must remain
    unchanged relative to the reference image.
    On failure, attempt to restore both original FoV and stage position.
    """
    center = np.asarray(center_m, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_m must contain two finite coordinates (x, y)")
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("diameter_m must be finite and positive")
    if not np.isfinite(padding) or padding < 1:
        raise ValueError("padding must be finite and at least 1")
    target_fov = float(diameter_m) * float(padding)
    if not np.isfinite(target_fov):
        raise ValueError("Requested FoV must be finite")
    if reference_stage_position_m is not None:
        reference = np.asarray(reference_stage_position_m, dtype=float)
        if reference.shape != (2,) or not np.all(np.isfinite(reference)):
            raise ValueError("reference_stage_position_m must contain finite (x, y)")
    original_fov = microscope.get_fov()
    original_stage = microscope.get_stage_position()
    if not np.isfinite(original_fov) or original_fov <= 0 or not np.all(np.isfinite(original_stage)):
        raise RuntimeError("Microscope returned an invalid original FoV or stage position")
    try:
        if reference_stage_position_m is None:
            microscope.center_fov_on_image_offset(float(center[0]), float(center[1]))
        else:
            microscope.center_fov_on_image_offset(float(center[0]), float(center[1]),
                reference_stage_position_m=tuple(reference))
        microscope.set_fov(target_fov)
        actual_fov = float(microscope.get_fov())
        actual_stage = microscope.get_stage_position()
        if not np.isfinite(actual_fov) or actual_fov <= 0 or not np.all(np.isfinite(actual_stage)):
            raise RuntimeError("Microscope returned an invalid FoV or stage position after positioning")
    except Exception as error:
        restore_errors = []
        for restore in (lambda: microscope.set_fov(original_fov),
                        lambda: microscope.set_stage_position(*original_stage)):
            try:
                restore()
            except Exception as restore_error:
                restore_errors.append(str(restore_error))
        if restore_errors:
            raise RuntimeError("Zoom2Fit failed and restoration was incomplete: " + "; ".join(restore_errors)) from error
        raise
    return Zoom2FitResult(
        (float(center[0]), float(center[1])), float(diameter_m), target_fov,
        actual_fov, original_stage, actual_stage,
    )
