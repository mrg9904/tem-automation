"""Acquire a current-position Ronchigram; orientation truth only for validation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import numpy as np
from nion.swift import Facade
from nion.swift.test import TestContext
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from algorithms.FindZone import find_zone_in_image


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-image', action='store_true')
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent/'find_zone_results/preview'
    if arguments.reuse_image:
        parameters = json.loads((root/'calibration.json').read_text())
        report = find_zone_in_image(np.load(root/'ronchigram.npy'), cif_path=parameters['cif_path'],
            calibration=parameters, output_directory=root, progress=lambda message: print(message, flush=True))
        print([(item['id'], item['correlation'], item['plausible']) for item in report['candidates']], flush=True)
        validate_against_saved_model(report, root)
        return
    setup = TestContext.TestSetup()
    configuration = DeviceConfiguration.AcquisitionContextConfiguration(sample_index=9, set_configuration_location=False)
    with tempfile.TemporaryDirectory() as settings:
        configuration.configuration_location = settings
        with AcquisitionTestContext.AcquisitionTestContext(configuration) as context:
            api = SimpleNamespace(get_instrument_by_id=lambda *a, **k: Facade.Instrument(context.instrument),
                get_hardware_source_by_id=lambda name, **k: Facade.HardwareSource(
                    context.camera_hardware_source if name == 'usim_ronchigram_camera' else context.scan_hardware_source))
            root = Path(__file__).resolve().parent/'find_zone_results/preview'
            root.mkdir(parents=True, exist_ok=True)
            with NionUSimRonchigramAdapter(api, timeout_s=120.) as microscope:
                print('Acquiring Ronchigram...', flush=True)
                image = microscope.acquire_ronchigram(exposure_ms=1000., binning=2)
                parameters = microscope.zone_capture_parameters()
                np.save(root/'ronchigram.npy', image)
                (root/'calibration.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
                print(parameters, flush=True)
            print(root, flush=True)


def validate_against_saved_model(report, root):
    """Independent audit only; never pass this data to the indexing algorithm."""
    from scipy.spatial.transform import Rotation
    from algorithms.crystallography import load_crystal_geometry
    from nion.usim_device import SampleGeometry
    from nion.utils import Geometry
    records_path = Path(__file__).resolve().parents[2]/'nionswift-usim/nion/usim_device/samples/1000_cathodes.npz'
    if not records_path.exists():
        return
    crystal = load_crystal_geometry(report['cif_path'])
    point = -np.asarray(report['calibration']['stage_xy_m'])*1e9
    tx, ty = report['calibration'].get('stage_tilt_xy_rad', [0., 0.])
    stage = SampleGeometry.stage_rotation(Geometry.FloatPoint(x=tx, y=ty))
    with np.load(records_path, allow_pickle=False) as records:
        vertices, rotations = records['vertices_lab_nm'], records['rotation_matrix']
        pivot = vertices[0].mean(axis=0)
        vertices = (vertices-pivot)@stage.T+pivot
        indices = np.flatnonzero(((vertices[:, :, :2].min(axis=1) <= point) &
                                 (vertices[:, :, :2].max(axis=1) >= point)).all(axis=1))
        checks = []
        for index in indices:
            for candidate in report['candidates']:
                error = min(np.rad2deg(Rotation.from_matrix(np.asarray(candidate['orientation_matrix']) @
                    symmetry @ (stage@rotations[index]).T).magnitude()) for symmetry in crystal.symmetries)
                checks.append(dict(candidate_id=candidate['id'], simulator_particle_id=int(index)+1,
                                   orientation_error_deg=float(error)))
    (root/'independent_validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('Independent validation:', checks, flush=True)
    if len(indices) == 1 and checks and report['candidates'][0]['correlation'] >= .8:
        best = next(item for item in checks if item['candidate_id'] == report['candidates'][0]['id'])
        assert best['orientation_error_deg'] < 1., best


if __name__ == '__main__':
    main()
