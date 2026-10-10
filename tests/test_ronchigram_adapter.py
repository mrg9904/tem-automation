"""Exercise the real Nion facade camera path as used by the importable script."""
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
try:
    from nion.swift import Facade
    from nion.swift.test import TestContext
    from nion.instrumentation.test import AcquisitionTestContext
    from nion.usim_device import DeviceConfiguration
except ImportError:
    Facade = None
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from algorithms.FindWindow import find_window_in_image, make_window_anchor


@unittest.skipIf(Facade is None, 'Optional integration test requires nionswift-dev and uSim')
class RonchigramAdapterTest(unittest.TestCase):
    def test_real_camera_acquisition_and_native_stage_mapping(self):
        setup = TestContext.TestSetup()
        configuration = DeviceConfiguration.AcquisitionContextConfiguration(set_configuration_location=False)
        with tempfile.TemporaryDirectory() as settings:
            configuration.configuration_location = settings
            self._check_camera(configuration)

    def _check_camera(self, configuration):
        with AcquisitionTestContext.AcquisitionTestContext(configuration) as context:
            api = SimpleNamespace(
                get_instrument_by_id=lambda *args, **kwargs: Facade.Instrument(context.instrument),
                get_hardware_source_by_id=lambda name, **kwargs: Facade.HardwareSource(
                    context.camera_hardware_source if name == 'usim_ronchigram_camera' else context.scan_hardware_source))
            with NionUSimRonchigramAdapter(api, timeout_s=120.) as microscope:
                microscope.initialize_window_view()
                self.assertEqual(context.instrument.scan_data_generator.sample.title, '1000CathodeParticleOnCarbon')
                np.testing.assert_allclose(microscope.get_stage_position(), [938e-9, -6820e-9])
                self.assertAlmostEqual(context.instrument.GetVal('stage_z_m'), 600e-6)
                image = microscope.acquire_ronchigram()
                result = find_window_in_image(image)
                anchor = make_window_anchor(result, *microscope.window_coordinates(
                    np.vstack((result.center_xy_px, result.corners_xy_px))))
                # These expectations validate the result; they are never algorithm inputs.
                self.assertAlmostEqual(anchor['mean_side_m']*1e6, 54., delta=.7)
                self.assertAlmostEqual(anchor['detector_angle_deg'], 60., delta=.4)
                np.testing.assert_allclose(anchor['center_stage_xy_m'], [0, 0], atol=.4e-6)
                microscope.set_stage_position(0., 0.)
                with self.assertRaisesRegex(RuntimeError, 'View changed'):
                    microscope.window_coordinates(result.corners_xy_px)
