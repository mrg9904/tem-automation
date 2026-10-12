"""Real uSim workflow preview, saved under tests/find_window_results."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
from nion.swift import Facade
from nion.swift.test import TestContext
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from scripts.usim_window_particle_workflow import run_window_workflow


def main():
    setup = TestContext.TestSetup()
    configuration = DeviceConfiguration.AcquisitionContextConfiguration(set_configuration_location=False)
    with tempfile.TemporaryDirectory() as settings:
        configuration.configuration_location = settings
        with AcquisitionTestContext.AcquisitionTestContext(configuration) as context:
            api = SimpleNamespace(
                get_instrument_by_id=lambda *args, **kwargs: Facade.Instrument(context.instrument),
                get_hardware_source_by_id=lambda name, **kwargs: Facade.HardwareSource(
                    context.camera_hardware_source if name == 'usim_ronchigram_camera' else context.scan_hardware_source))
            output = Path(__file__).resolve().parent/'find_window_results/preview'
            with NionUSimRonchigramAdapter(api, timeout_s=120.) as microscope:
                print(run_window_workflow(microscope, output))
            print(output)


if __name__ == '__main__':
    main()
