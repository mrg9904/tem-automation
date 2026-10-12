"""Reproducible timing with either a saved capture or an isolated tilted uSim."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import tempfile
from types import SimpleNamespace
import numpy as np
from algorithms.FindZone import find_zone_in_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tilt', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--track-roi', action='store_true', help='Test-only geometric position compensation for sample tilt')
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    if arguments.reuse_directory:
        source = Path(arguments.reuse_directory)
        calibration = json.loads((source/'find_zone.json').read_text())['calibration']
        with np.load(source/'find_zone.npz', allow_pickle=False) as saved:
            image = saved['image'].copy()
        run(image, calibration, output, 0., perf_counter())
    else:
        from nion.swift import Facade
        from nion.swift.test import TestContext
        from nion.instrumentation.test import AcquisitionTestContext
        from nion.usim_device import DeviceConfiguration
        from nion.utils import Geometry
        from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
        setup = TestContext.TestSetup()
        configuration = DeviceConfiguration.AcquisitionContextConfiguration(sample_index=9, set_configuration_location=False)
        with tempfile.TemporaryDirectory() as settings:
            configuration.configuration_location = settings
            with AcquisitionTestContext.AcquisitionTestContext(configuration) as context:
                sample = context.instrument.scan_data_generator.sample
                tracked_point = None
                if arguments.track_roi:
                    stage = context.instrument.stage_position_m
                    x, y = -stage.x*1e9, -stage.y*1e9
                    for index in sample.visible_indices(np.asarray([x]), np.asarray([y])):
                        crystal = sample.crystal(int(index))
                        lo, hi = crystal._crystal.surface_maps(np.asarray([x]), np.asarray([y]))
                        if np.isfinite(lo).all() and np.isfinite(hi).all() and float((hi-lo).sum()) > 0:
                            tracked_point = np.array([x, y, float((hi+lo).mean()/2)])
                            break
                    if tracked_point is None:
                        raise ValueError('No material point found for geometric ROI tracking')
                context.instrument.SetVal2D('stage_tilt_rad', Geometry.FloatPoint(x=np.deg2rad(arguments.tilt[0]),
                                                                                 y=np.deg2rad(arguments.tilt[1])))
                if tracked_point is not None:
                    from nion.usim_device import SampleGeometry
                    transform = SampleGeometry.stage_rotation(Geometry.FloatPoint(x=np.deg2rad(arguments.tilt[0]),
                                                                                 y=np.deg2rad(arguments.tilt[1])))
                    point = transform@(tracked_point-sample.center_nm)+sample.center_nm
                    context.instrument.SetVal2D('stage_position_m', Geometry.FloatPoint(x=-point[0]*1e-9, y=-point[1]*1e-9))
                    print('Test ROI tracked to Stage XY (nm):', (-point[:2]).tolist(), flush=True)
                api = SimpleNamespace(get_instrument_by_id=lambda *a, **k: Facade.Instrument(context.instrument),
                    get_hardware_source_by_id=lambda name, **k: Facade.HardwareSource(
                        context.camera_hardware_source if name == 'usim_ronchigram_camera' else context.scan_hardware_source))
                with NionUSimRonchigramAdapter(api, timeout_s=120.) as microscope:
                    started = perf_counter()
                    image = microscope.acquire_ronchigram(exposure_ms=1000., binning=2)
                    calibration = microscope.zone_capture_parameters()
                    acquisition = perf_counter()-started
                    run(image, calibration, output, acquisition, started)


def run(image, calibration, output, acquisition, started):
    markers = []
    def progress(message):
        markers.append(dict(elapsed_seconds=perf_counter()-started, message=message))
        print(f"[{markers[-1]['elapsed_seconds']:.2f}s] {message}", flush=True)
    report = find_zone_in_image(image, cif_path=calibration['cif_path'], calibration=calibration,
                               output_directory=output, progress=progress)
    total = perf_counter()-started
    (output/'timings.json').write_text(json.dumps(dict(acquisition_seconds=acquisition, total_seconds=total,
                                                     progress=markers), indent=2), encoding='utf-8')
    print('Total seconds:', total, flush=True)
    for candidate in report['candidates'][:2]:
        print(candidate['id'], candidate['correlation'], candidate['nearest_small_index_zones'][0], flush=True)


if __name__ == '__main__':
    main()
