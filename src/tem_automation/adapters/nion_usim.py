from __future__ import annotations

import time
from typing import Any

import numpy as np

from tem_automation.core.exceptions import AcquisitionError
from tem_automation.core.exceptions import ConnectionError
from tem_automation.core.exceptions import MicroscopeError
from tem_automation.core.models import AcquisitionSettings
from tem_automation.core.models import ImageFrame
from tem_automation.core.models import ImagingMode
from tem_automation.core.models import MicroscopeState


class NionUSimAdapter:
    """Nion Swift facade adapter for the uSim STEM scan device."""

    def __init__(
        self,
        api: Any,
        *,
        instrument_id: str = "usim_stem_controller",
        hardware_source_id: str = "usim_scan_device",
        profile_index: int = 0,
    ) -> None:
        self._api = api
        self._instrument_id = instrument_id
        self._hardware_source_id = hardware_source_id
        self._profile_index = profile_index
        self._instrument: Any | None = None
        self._hardware_source: Any | None = None

    def connect(self) -> None:
        self._instrument = self._api.get_instrument_by_id(
            self._instrument_id,
            version="~1.0",
        )
        self._hardware_source = self._api.get_hardware_source_by_id(
            self._hardware_source_id,
            version="~1.0",
        )

        if self._instrument is None:
            raise ConnectionError(
                f"Nion instrument not found: {self._instrument_id}"
            )
        if self._hardware_source is None:
            raise ConnectionError(
                f"Nion hardware source not found: {self._hardware_source_id}"
            )

    def close(self) -> None:
        self._instrument = None
        self._hardware_source = None

    def get_state(self) -> MicroscopeState:
        instrument = self._require_instrument()
        return MicroscopeState(
            defocus_m=float(instrument.get_control_output("C10")),
        )

    def set_defocus(self, defocus_m: float) -> None:
        instrument = self._require_instrument()
        try:
            instrument.set_control_output("C10", float(defocus_m))
        except Exception as exc:
            raise MicroscopeError(
                f"uSim rejected defocus {defocus_m:.6e} m"
            ) from exc

    def acquire(self, settings: AcquisitionSettings) -> ImageFrame:
        if settings.mode is not ImagingMode.HAADF_STEM:
            raise AcquisitionError(
                f"NionUSimAdapter does not support mode {settings.mode.value}"
            )

        hardware_source = self._require_hardware_source()
        frame_parameters = (
            hardware_source.get_frame_parameters_for_profile_by_index(
                self._profile_index
            )
        )
        frame_parameters["pixel_size"] = (
            settings.height_px,
            settings.width_px,
        )
        frame_parameters["fov_nm"] = settings.fov_m * 1e9
        frame_parameters["pixel_time_us"] = settings.dwell_time_s * 1e6

        try:
            frames = hardware_source.record(
                frame_parameters,
                None,
                settings.timeout_s,
            )
        except Exception as exc:
            raise AcquisitionError("uSim HAADF acquisition failed") from exc

        if not frames:
            raise AcquisitionError("uSim returned no HAADF frame")

        xdata = frames[0]
        data = np.asarray(xdata.data, dtype=np.float32)
        if data.ndim != 2:
            raise AcquisitionError(
                f"Expected a 2-D HAADF image, received shape {data.shape}"
            )

        metadata = dict(getattr(xdata, "metadata", {}) or {})
        metadata["adapter"] = "nion_usim"
        metadata["defocus_m"] = self.get_state().defocus_m

        return ImageFrame(
            data=data,
            pixel_size_m=(
                settings.fov_m / settings.height_px,
                settings.fov_m / settings.width_px,
            ),
            timestamp_s=time.time(),
            metadata=metadata,
        )

    def _require_instrument(self) -> Any:
        if self._instrument is None:
            raise ConnectionError("NionUSimAdapter is not connected")
        return self._instrument

    def _require_hardware_source(self) -> Any:
        if self._hardware_source is None:
            raise ConnectionError("NionUSimAdapter is not connected")
        return self._hardware_source

