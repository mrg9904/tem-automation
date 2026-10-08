import unittest

import numpy as np

from adapters.nion_usim import NionUSimAdapter


class _FakeInstrument:
    def __init__(self) -> None:
        self.controls = {"C10": 50e-9, "stage_position_m.x": 1222e-9, "stage_position_m.y": 279e-9}

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
        self.profile_index = 0
        self.is_recording = False
        self.current_parameters = {
            "fov_nm": 80.0,
        }

    def get_frame_parameters_for_profile_by_index(self, profile_index):
        return dict(self.current_parameters)

    def set_frame_parameters_for_profile_by_index(self, profile_index, frame_parameters):
        self.last_profile_index = profile_index
        self.current_parameters = dict(frame_parameters)

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
        adapter = NionUSimAdapter(
        api,
        image_size_px=48,
        dwell_time_us=2.0,
    )
        adapter.connect()
        self.assertAlmostEqual(adapter.get_fov(), 80e-9)
        adapter.set_defocus(125e-9)

        image = adapter.acquire_haadf()

        self.assertAlmostEqual(api.instrument.controls["C10"], 125e-9)
        self.assertEqual(
            api.hardware_source.last_parameters["pixel_size"],
            (48, 48),
        )
        self.assertAlmostEqual(
            api.hardware_source.last_parameters["fov_nm"],
            80.0,
        )
        self.assertAlmostEqual(
            api.hardware_source.last_parameters["pixel_time_us"],
            2.0,
        )
        self.assertEqual(image.shape, (32, 48))


    def test_zoom_updates_selected_profile_preserving_parameters_and_focus(self):
        from algorithms.Zoom2Fit import zoom_to_fit
        api = _FakeAPI()
        api.hardware_source.profile_index = 1
        api.hardware_source.current_parameters.update(rotation_rad=0.0, pixel_time_us=3.0)
        with NionUSimAdapter(api) as microscope:
            result = zoom_to_fit(microscope, (100e-9, -50e-9), 200e-9)
            np.testing.assert_allclose(microscope.get_stage_position(), (1122e-9, 329e-9))
            self.assertAlmostEqual(result.actual_fov_m, 200e-9, delta=1e-18)
            self.assertEqual(microscope.get_defocus(), 50e-9)
            self.assertEqual(api.hardware_source.last_profile_index, 1)
            self.assertEqual(api.hardware_source.current_parameters['pixel_time_us'], 3.0)
            microscope.acquire_haadf()
            self.assertAlmostEqual(api.hardware_source.last_parameters['fov_nm'], 200.0)

    def test_centering_applies_scan_rotation(self):
        api = _FakeAPI()
        api.hardware_source.current_parameters['rotation_rad'] = np.pi / 2
        with NionUSimAdapter(api) as microscope:
            microscope.center_fov_on_image_offset(100e-9, 0.0)
            np.testing.assert_allclose(microscope.get_stage_position(), (1222e-9, 179e-9))

    def test_temporary_focus_resolution_restores_final_capture_after_failure(self):
        api = _FakeAPI()
        with NionUSimAdapter(api, image_size_px=1024) as microscope:
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with microscope.acquisition_settings(image_size_px=512):
                    microscope.acquire_haadf()
                    self.assertEqual(api.hardware_source.last_parameters['pixel_size'], (512, 512))
                    raise RuntimeError('test failure')
            microscope.acquire_haadf()
            self.assertEqual(api.hardware_source.last_parameters['pixel_size'], (1024, 1024))

    def test_reference_positioning_moves_directly_and_handles_rotation(self):
        from algorithms.Zoom2Fit import zoom_to_fit
        api = _FakeAPI()
        api.hardware_source.current_parameters['rotation_rad'] = np.pi / 2
        with NionUSimAdapter(api) as microscope:
            reference = microscope.get_stage_position()
            zoom_to_fit(microscope, (100e-9, 0), 50e-9, reference_stage_position_m=reference)
            zoom_to_fit(microscope, (100e-9, 50e-9), 50e-9, reference_stage_position_m=reference)
            np.testing.assert_allclose(microscope.get_stage_position(), (1272e-9, 179e-9))

    def test_final_capture_profile_survives_temporary_settings_and_later_focus_scan(self):
        api = _FakeAPI()
        with NionUSimAdapter(api, image_size_px=512) as microscope:
            with microscope.acquisition_settings(image_size_px=1024, dwell_time_us=1.):
                microscope.set_fov(50e-9)
                microscope.acquire_haadf()
            final = microscope.get_last_scan_profile()
            microscope.set_fov(800e-9)
            microscope.acquire_haadf()
            microscope.persist_last_scan_profile(final)
            self.assertEqual(api.hardware_source.current_parameters['pixel_size'],(1024,1024))
            self.assertAlmostEqual(api.hardware_source.current_parameters['fov_nm'],50.)
            self.assertEqual(api.hardware_source.current_parameters['pixel_time_us'],1.)


if __name__ == "__main__":
    unittest.main()
