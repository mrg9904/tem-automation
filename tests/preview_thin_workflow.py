"""Run the complete window-to-EELS workflow using actual uSim sources."""
from pathlib import Path
import argparse
import json
import numpy as np
from types import SimpleNamespace
import tempfile
from nion.swift import Facade
from nion.swift.test import TestContext
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration
from scripts.usim_thin_workflow import run_thin_workflow
from scripts.window_particle_workflow import run_window_workflow
from scripts.thin_particle_workflow import map_particles
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from adapters.nion_usim_eels import NionUSimEELSAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Preview only: limit real particle mappings')
    parser.add_argument('--reuse-window', action='store_true', help='Preview only: reuse the saved calibrated window')
    arguments = parser.parse_args()
    setup = TestContext.TestSetup()
    configuration = DeviceConfiguration.AcquisitionContextConfiguration(sample_index=9, set_configuration_location=False)
    with tempfile.TemporaryDirectory() as settings:
        configuration.configuration_location = settings
        with AcquisitionTestContext.AcquisitionTestContext(configuration) as context:
            api = SimpleNamespace(
                get_instrument_by_id=lambda *args, **kwargs: Facade.Instrument(context.instrument),
                get_hardware_source_by_id=lambda name, **kwargs: Facade.HardwareSource(
                    context.camera_hardware_source if name == 'usim_ronchigram_camera' else context.scan_hardware_source))
            output = Path(__file__).resolve().parent/'find_thin_results/batch_workflow_preview'
            progress = lambda message: print(message, flush=True)
            if arguments.limit is None:
                report = run_thin_workflow(api, output, progress=progress)
            else:
                if arguments.limit < 1:
                    raise ValueError('Preview limit must be positive')
                if arguments.reuse_window:
                    window = json.loads((output/'window/workflow.json').read_text())
                else:
                    with NionUSimRonchigramAdapter(api, timeout_s=120.) as microscope:
                        window = run_window_workflow(microscope, output/'window', progress=progress, zoom_first_particle=False)
                with np.load(output/'window/cell_0001/find_particles.npz', allow_pickle=False) as saved:
                    particles = saved['particles'][:arguments.limit].copy()
                    statuses = saved['center_status'][:arguments.limit].copy()
                with NionUSimEELSAdapter(api, timeout_s=120.) as microscope:
                    if arguments.reuse_window:
                        microscope._require_instrument().set_control_output('stage_z_m', 0.)
                        microscope.set_defocus(0.)
                        parameters = microscope._get_active_profile_frame_parameters()
                        parameters.update(rotation_rad=window['cell']['scan_rotation_rad'], center_nm=(0., 0.))
                        microscope._require_hardware_source().set_frame_parameters_for_profile_by_index(0, parameters)
                    report = map_particles(microscope, particles, statuses, window['cell_reference_stage_xy_m'],
                        output, window_report=window, progress=progress)
            assert report['status'] in ('complete', 'complete_with_issues')
            assert report['ranking'], 'No actual particle mapping succeeded'
            assert context.instrument.GetVal('stage_z_m') == 0.
            assert all(np.isclose(entry['mapping_lens_defocus_m'],
                entry['autofocus']['best_defocus_m']) for entry in report['particles']
                if 'mapping_lens_defocus_m' in entry)
            assert (output/report['ranking'][0]['directory']/'thickness/haadf_thin_annotated.png').exists()
            print(output, flush=True)


if __name__ == '__main__':
    main()
