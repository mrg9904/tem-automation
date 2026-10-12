"""Full 128x128 uSim spectrum-image validation at the last particle target."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from nion.swift import Facade
from nion.swift.test import TestContext
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration
from adapters.nion_usim_eels import NionUSimEELSAdapter
from algorithms.FindThin import find_thin
from algorithms.AutoBC import auto_bc


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root/'find_window_results/preview/workflow.json').read_text())
    setup = TestContext.TestSetup()
    configuration = DeviceConfiguration.AcquisitionContextConfiguration(sample_index=9, set_configuration_location=False)
    with tempfile.TemporaryDirectory() as settings:
        configuration.configuration_location = settings
        with AcquisitionTestContext.AcquisitionTestContext(configuration) as context:
            api = SimpleNamespace(
                get_instrument_by_id=lambda *args, **kwargs: Facade.Instrument(context.instrument),
                get_hardware_source_by_id=lambda name, **kwargs: Facade.HardwareSource(context.scan_hardware_source))
            with NionUSimEELSAdapter(api, timeout_s=120.) as microscope:
                microscope.set_stage_position(*report['particle_zoom']['actual_stage_position_m'])
                microscope.set_fov(report['particle_zoom']['actual_fov_m'])
                parameters = context.scan_hardware_source.get_frame_parameters(0)
                parameters.rotation_rad = report['cell']['scan_rotation_rad']
                context.scan_hardware_source.set_frame_parameters(0, parameters)
                output = root/'find_thin_results/preview_128'
                result = find_thin(microscope, output_directory=output)
                microscope.persist_last_scan_profile()
                tone = auto_bc(result.haadf, result.particle_mask)
                microscope.set_display_tone(tone.brightness, tone.contrast)
                saved = microscope._get_active_profile_frame_parameters()
                assert saved['usim_brightness'] == tone.brightness
                assert saved['usim_contrast'] == tone.contrast
                print('AutoBC scan profile verified:', tone.metadata())
                print(result.summary())
                print(output)


if __name__ == '__main__':
    main()
