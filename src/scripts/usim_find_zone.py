"""Nion Swift: index the current Ronchigram without changing stage or focus."""
from pathlib import Path
from datetime import datetime
from time import monotonic
from uuid import uuid4
from scripts import __file__ as scripts_package_file
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from adapters.nion_usim import nion_cancel_callback
from adapters.cancellation import cancellation_scope, WorkflowCancelled
from algorithms.FindZone import find_zone_in_image, FindZoneConfig

CONFIG = FindZoneConfig()
CIF_PATH = None  # Set a path for another known phase; default reads sample CIF.


def run_find_zone(microscope, directory, *, cif_path=None, config=CONFIG, progress=print):
    started = monotonic()
    def notify(message):
        progress(f'[{monotonic()-started:.1f}s] {message}')
    notify('Acquiring current-position Ronchigram...')
    image = microscope.acquire_ronchigram(exposure_ms=1000., binning=2)
    calibration = microscope.zone_capture_parameters()
    if not calibration.get('kikuchi_enabled', True):
        raise ValueError('Kikuchi calculation is disabled at this Stage Z; use a smaller Stage Z')
    path = cif_path or calibration.get('cif_path')
    if not path:
        raise ValueError('Provide the phase CIF file for FindZone')
    result = find_zone_in_image(image, cif_path=path, calibration=calibration,
                               output_directory=directory, config=config, progress=notify)
    notify(f"FindZone {result['status']}: {len(result['candidates'])} orientation candidates.")
    for candidate in result['candidates']:
        if candidate['plausible']:
            notify(f"Candidate {candidate['id']}: image correlation {candidate['correlation']:.3f}; "
                   f"beam direction proportional to {candidate['beam_direction_uvw_proportional']}.")
            nearest = candidate['nearest_small_index_zones'][0]
            notify(f"Nearest small-index zone: {nearest['zone_axis']}, "
                   f"axis angle {nearest['beam_axis_angle_deg']:.3f} deg; equivalents {nearest['equivalent_zone_axes']}.")
    for row in sorted(result['tilt_suggestions'], key=lambda row: row['beam_axis_angle_deg'])[:8]:
        progress(f"{row['candidate_id']}: {'/'.join(row['possible_zone_axes'])}, "
                 f"axis angle {row['beam_axis_angle_deg']:.2f} deg, "
                 f"delta TX {row['delta_tx_deg']:+.2f} deg, delta TY {row['delta_ty_deg']:+.2f} deg")
    return result


def script_main(api_broker):
    directory = Path(scripts_package_file).resolve().parent/'find_zone_results'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8])
    try:
        with cancellation_scope(nion_cancel_callback(print)):
            with NionUSimRonchigramAdapter(api_broker.get_api(version='~1.0'), timeout_s=120.) as microscope:
                run_find_zone(microscope, directory, cif_path=CIF_PATH)
    except WorkflowCancelled:
        print('FindZone cancelled.')
    except Exception as error:
        print(f'***Warning*** FindZone failed: {error}')
    finally:
        print(f'Images and data: {directory}')
