from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

class NionUSimAdapter:
    """Nion Swift facade adapter for the uSim STEM scan device."""

    def __init__(
        self,
        api: Any,
        *,
        instrument_id: str = "usim_stem_controller",
        hardware_source_id: str = "usim_scan_device",
        profile_index: int = 0,
        fov_nm: float = 100.0,
        image_size_px: int = 256,
        dwell_time_us: float = 1.0,
        timeout_s: float = 30.0,
    ) -> None:
        if fov_nm <= 0:
            raise ValueError("fov_nm must be positive")
        if image_size_px <= 1:
            raise ValueError("image_size_px must exceed one pixel")
        if dwell_time_us <= 0:
            raise ValueError("dwell_time_us must be positive")

        self._api = api
        self._instrument_id = instrument_id
        self._hardware_source_id = hardware_source_id
        self._profile_index = profile_index
        self._fov_nm = fov_nm
        self._image_size_px = image_size_px
        self._dwell_time_us = dwell_time_us
        self._timeout_s = timeout_s
        self._instrument: Any | None = None
        self._hardware_source: Any | None = None

    def __enter__(self) -> NionUSimAdapter:
        self.connect()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

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
            raise RuntimeError(
                f"Nion instrument not found: {self._instrument_id}"
            )
        if self._hardware_source is None:
            raise RuntimeError(
                f"Nion hardware source not found: {self._hardware_source_id}"
            )

    def close(self) -> None:
        self._instrument = None
        self._hardware_source = None

    def get_defocus(self) -> float:
        instrument = self._require_instrument()
        return float(instrument.get_control_output("C10"))

    def set_defocus(self, defocus_m: float) -> None:
        instrument = self._require_instrument()
        try:
            instrument.set_control_output("C10", float(defocus_m))
        except Exception as exc:
            raise RuntimeError(
                f"uSim rejected defocus {defocus_m:.6e} m"
            ) from exc

    def get_fov(self) -> float:
        """Return the acquisition field of view in meters."""
        return float(self._fov_nm) * 1e-9

    def acquire_haadf(self) -> npt.NDArray[np.float32]:
        hardware_source = self._require_hardware_source()
        frame_parameters = (
            hardware_source.get_frame_parameters_for_profile_by_index(
                self._profile_index
            )
        )
        frame_parameters["pixel_size"] = (
            self._image_size_px,
            self._image_size_px,
        )
        frame_parameters["fov_nm"] = self._fov_nm
        frame_parameters["pixel_time_us"] = self._dwell_time_us

        try:
            frames = hardware_source.record(
                frame_parameters,
                None,
                self._timeout_s,
            )
        except Exception as exc:
            raise RuntimeError("uSim HAADF acquisition failed") from exc

        if not frames:
            raise RuntimeError("uSim returned no HAADF frame")

        xdata = frames[0]
        data = np.asarray(xdata.data, dtype=np.float32)
        if data.ndim != 2:
            raise RuntimeError(
                f"Expected a 2-D HAADF image, received shape {data.shape}"
            )
        return data

    def _require_instrument(self) -> Any:
        if self._instrument is None:
            raise RuntimeError("NionUSimAdapter is not connected")
        return self._instrument

    def _require_hardware_source(self) -> Any:
        if self._hardware_source is None:
            raise RuntimeError("NionUSimAdapter is not connected")
        return self._hardware_source
