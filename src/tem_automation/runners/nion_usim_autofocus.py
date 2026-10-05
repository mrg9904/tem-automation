from __future__ import annotations

from tem_automation.adapters.nion_usim import NionUSimAdapter
from tem_automation.core.models import AcquisitionSettings
from tem_automation.experiments.autofocus import AutofocusResult
from tem_automation.experiments.autofocus import autofocus


def run(
    *,
    search_half_range_nm: float = 200.0,
    coarse_points: int = 9,
    fine_points: int = 7,
    fov_nm: float = 100.0,
    image_size_px: int = 256,
    dwell_time_us: float = 1.0,
    settle_time_s: float = 0.0,
    frames_per_position: int = 1,
) -> AutofocusResult:
    """Run platform-independent autofocus using the active Nion uSim."""
    from nion.swift import Facade

    api = Facade.get_api("~1.0", "~1.0")
    microscope = NionUSimAdapter(api)
    microscope.connect()
    try:
        result = autofocus(
            microscope,
            AcquisitionSettings(
                fov_m=fov_nm * 1e-9,
                width_px=image_size_px,
                height_px=image_size_px,
                dwell_time_s=dwell_time_us * 1e-6,
            ),
            search_half_range_m=search_half_range_nm * 1e-9,
            coarse_points=coarse_points,
            fine_points=fine_points,
            settle_time_s=settle_time_s,
            frames_per_position=frames_per_position,
        )
    finally:
        microscope.close()

    return result

