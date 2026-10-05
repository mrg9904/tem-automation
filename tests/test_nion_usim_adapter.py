import unittest

import numpy as np

from tem_automation.adapters.nion_usim import NionUSimAdapter
from tem_automation.core.models import AcquisitionSettings


class _FakeInstrument:
    def __init__(self) -> None:
        self.controls = {"C10": 50e-9}

    def get_control_output(self, name: str) -> float:
        return self.controls[name]

    def set_control_output(self, name: str, value: float) -> None:
        self.controls[name] = value


class _FakeXData:
    def __init__(self) -> None:
        self.data = np.ones((32, 48), dtype=np.float32)
        self.metadata = {"hardware_source": {"id": "usim_scan_device"}}


class _FakeHardwareSource:
    def __init__(self) -> None:
        self.last_parameters = None

    def get_frame_parameters_for_profile_by_index(self, index: int):
        return {"profile_index": index}

    def record(self, frame_parameters, channels_enabled, timeout):
        self.last_parameters = frame_parameters
        return [_FakeXData()]


class _FakeAPI:
    def __init__(self) -> None:
        self.instrument = _FakeInstrument()
        self.hardware_source = _FakeHardwareSource()

    def get_instrument_by_id(self, instrument_id: str, version: str):
        return self.instrument

    def get_hardware_source_by_id(self, hardware_source_id: str, version: str):
        return self.hardware_source


class NionUSimAdapterTest(unittest.TestCase):
    def test_adapter_translates_si_units_to_usim_frame_parameters(self) -> None:
        api = _FakeAPI()
        adapter = NionUSimAdapter(api)
        adapter.connect()
        adapter.set_defocus(125e-9)

        frame = adapter.acquire(
            AcquisitionSettings(
                fov_m=80e-9,
                width_px=48,
                height_px=32,
                dwell_time_s=2e-6,
            )
        )

        self.assertAlmostEqual(api.instrument.controls["C10"], 125e-9)
        self.assertEqual(
            api.hardware_source.last_parameters["pixel_size"],
            (32, 48),
        )
        self.assertAlmostEqual(
            api.hardware_source.last_parameters["fov_nm"],
            80.0,
        )
        self.assertAlmostEqual(
            api.hardware_source.last_parameters["pixel_time_us"],
            2.0,
        )
        self.assertEqual(frame.data.shape, (32, 48))


if __name__ == "__main__":
    unittest.main()
