from __future__ import annotations

import time
import threading
from contextlib import contextmanager

from adapters.cancellation import check_cancelled, suspend_cancellation
from typing import Any
from adapters.coordinates import ImageStageTransform

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
        image_size_px: int = 256,
        dwell_time_us: float = 1.0,
        timeout_s: float = 30.0,
    ) -> None:
        if image_size_px <= 1:
            raise ValueError("image_size_px must exceed one pixel")
        if dwell_time_us <= 0:
            raise ValueError("dwell_time_us must be positive")

        self._api = api
        self._instrument_id = instrument_id
        self._hardware_source_id = hardware_source_id
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

    @contextmanager
    def acquisition_settings(self, *, image_size_px, dwell_time_us=1.0):
        """Temporarily lower acquisition cost, restoring final capture settings."""
        if image_size_px <= 1 or not np.isfinite(dwell_time_us) or dwell_time_us <= 0:
            raise ValueError("Invalid temporary acquisition settings")
        previous = self._image_size_px, self._dwell_time_us
        self._image_size_px, self._dwell_time_us = image_size_px, dwell_time_us
        try:
            yield
        finally:
            self._image_size_px, self._dwell_time_us = previous

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
        """Return the currently selected scan profile FoV in meters."""
        frame_parameters = self._get_active_profile_frame_parameters()
        try:
            fov_nm = float(frame_parameters["fov_nm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "The active uSim scan profile has no valid FoV"
            ) from exc
        if not np.isfinite(fov_nm) or fov_nm <= 0:
            raise RuntimeError(
                f"The active uSim scan profile has invalid FoV {fov_nm!r} nm"
            )
        return fov_nm * 1e-9

    def set_fov(self, fov_m: float) -> None:
        """Persist FoV in the selected scan profile, preserving other settings."""
        if not np.isfinite(fov_m) or fov_m <= 0:
            raise ValueError("fov_m must be finite and positive")
        source = self._require_hardware_source()
        self._wait_until_recording_finishes(source)
        profile_index = int(source.profile_index)
        parameters = dict(source.get_frame_parameters_for_profile_by_index(profile_index))
        parameters["fov_nm"] = float(fov_m) * 1e9
        source.set_frame_parameters_for_profile_by_index(profile_index, parameters)

    def get_stage_position(self) -> tuple[float, float]:
        instrument = self._require_instrument()
        return (float(instrument.get_control_output("stage_position_m.x")),
                float(instrument.get_control_output("stage_position_m.y")))

    def set_stage_position(self, x_m: float, y_m: float) -> None:
        if not np.all(np.isfinite([x_m, y_m])):
            raise ValueError("Stage coordinates must be finite")
        instrument = self._require_instrument()
        instrument.set_control_output("stage_position_m.x", float(x_m))
        instrument.set_control_output("stage_position_m.y", float(y_m))

    def get_last_scan_profile(self):
        parameters = getattr(self, '_last_record_parameters', None)
        return None if parameters is None else {'profile_index': self._last_record_profile_index,
                                                'parameters': dict(parameters)}

    def persist_last_scan_profile(self, snapshot=None):
        """Apply the last successful record parameters to the live scan profile."""
        snapshot = self.get_last_scan_profile() if snapshot is None else snapshot
        if snapshot is None:
            return None
        parameters = snapshot['parameters']
        source = self._require_hardware_source()
        self._wait_until_recording_finishes(source)
        index = snapshot['profile_index']
        source.set_frame_parameters_for_profile_by_index(index, dict(parameters))
        source.profile_index = index
        # Keep adapter settings consistent for any subsequent manual/script capture.
        self._image_size_px = int(parameters['pixel_size'][0])
        self._dwell_time_us = float(parameters['pixel_time_us'])
        return {'profile_index': index, 'parameters': dict(parameters)}

    def get_coordinate_transform(self):
        return ImageStageTransform(self.get_scan_rotation(), stage_direction=-1.)

    def get_scan_rotation(self) -> float:
        rotation = float(self._get_active_profile_frame_parameters().get("rotation_rad", 0.0))
        if not np.isfinite(rotation):
            raise RuntimeError("Scan rotation must be finite")
        return rotation

    def center_fov_on_image_offset(self, x_m: float, y_m: float, *,
                                   reference_stage_position_m: tuple[float, float] | None = None) -> None:
        """Match uSim's double-click centering, including scan rotation."""
        if not np.all(np.isfinite([x_m, y_m])):
            raise ValueError("Image offsets must be finite")
        reference = self.get_stage_position() if reference_stage_position_m is None else reference_stage_position_m
        target = self.get_coordinate_transform().stage_position_for_offset((x_m,y_m),reference)
        self.set_stage_position(*target)

    def acquire_haadf(self) -> npt.NDArray[np.float32]:
        hardware_source = self._require_hardware_source()
        frame_parameters = self._get_active_profile_frame_parameters()
        frame_parameters["pixel_size"] = (
            self._image_size_px,
            self._image_size_px,
        )
        frame_parameters["pixel_time_us"] = self._dwell_time_us

        check_cancelled()
        try:
            # Facade.record can return image data slightly before Nion has
            # completely torn down the previous record task. Wait for the
            # hardware source to become idle before starting another frame.
            self._wait_until_recording_finishes(hardware_source)
            frames = self._record_cancellable(hardware_source, frame_parameters)
            self._wait_until_recording_finishes(hardware_source)
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
        self._last_record_parameters = dict(frame_parameters)
        self._last_record_profile_index = int(hardware_source.profile_index)
        return data

    def _record_cancellable(self, source, parameters):
        pending = getattr(self, "_pending_record", None)
        if pending is not None and pending.is_alive():
            raise RuntimeError("Previous HAADF recording has not stopped; refusing another acquisition")
        finished = threading.Event()
        outcome = {}
        def record():
            try:
                outcome["frames"] = source.record(parameters, None, self._timeout_s)
            except BaseException as error:
                outcome["error"] = error
            finally:
                finished.set()
        worker = threading.Thread(target=record, daemon=True)
        self._pending_record = worker
        worker.start()
        deadline = time.monotonic() + self._timeout_s
        try:
            while not finished.wait(.05):
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("HAADF recording timed out")
            check_cancelled()
            if "error" in outcome:
                raise outcome["error"]
            return outcome["frames"]
        except BaseException:
            # Repeated abort handles cancellation racing recording startup.
            stop_deadline = time.monotonic() + 3.0
            while not finished.is_set():
                abort = getattr(source, "abort_recording", None)
                if callable(abort):
                    abort()
                if finished.wait(.05) or time.monotonic() >= stop_deadline:
                    break
            raise

    def _get_active_profile_frame_parameters(self) -> dict[str, Any]:
        """Return a copy of the currently selected scan profile settings."""
        hardware_source = self._require_hardware_source()
        profile_index = int(hardware_source.profile_index)
        return hardware_source.get_frame_parameters_for_profile_by_index(
            profile_index
        )

    def _wait_until_recording_finishes(
        self,
        hardware_source: Any,
    ) -> None:
        """Wait until Nion has completely released its record task."""
        deadline = time.monotonic() + self._timeout_s
        while bool(hardware_source.is_recording):
            check_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the uSim record task to finish"
                )
            time.sleep(0.01)

    def _require_instrument(self) -> Any:
        if self._instrument is None:
            raise RuntimeError("NionUSimAdapter is not connected")
        return self._instrument

    def _require_hardware_source(self) -> Any:
        if self._hardware_source is None:
            raise RuntimeError("NionUSimAdapter is not connected")
        return self._hardware_source


def nion_cancel_callback(print_function):
    # ScriptsDialog supplies its bound print method to the executed script.
    dialog = getattr(print_function, "__self__", None)
    return lambda: bool(getattr(dialog, "cancelled", False) or getattr(dialog, "_RunScriptDialog__is_closed", False))
