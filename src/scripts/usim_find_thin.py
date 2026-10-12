"""Nion Swift entry: map thin areas of the particle in the current scan FoV."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from scripts import __file__ as scripts_package_file
from adapters.nion_usim_eels import NionUSimEELSAdapter
from adapters.nion_usim import nion_cancel_callback
from adapters.cancellation import cancellation_scope, WorkflowCancelled
from algorithms.FindThin import find_thin, FindThinConfig
from algorithms.AutoBC import auto_bc

CONFIG = FindThinConfig(image_size_px=128, dispersion_eV=2., energy_max_eV=498.)


def script_main(api_broker):
    directory = Path(scripts_package_file).resolve().parent/'find_thin_results'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8])
    try:
        with cancellation_scope(nion_cancel_callback(print)):
            with NionUSimEELSAdapter(api_broker.get_api(version='~1.0'), timeout_s=120.) as microscope:
                result = find_thin(microscope, output_directory=directory, config=CONFIG, progress=print)
                microscope.persist_last_scan_profile()
                tone = auto_bc(result.haadf, result.particle_mask)
                microscope.set_display_tone(tone.brightness, tone.contrast)
        summary = result.summary()
        print(f"Thin area (t/lambda < 1): {summary['thin_area_percent']:.2f}%.")
        if summary['particle_touches_edge'] and not summary['has_complete_context_footprint']:
            print('***Warning*** Particle touches FoV boundary; percentage refers to visible particle area.')
        elif summary['area_outside_current_fov_percent'] > 0:
            print('***Warning*** EELS covers only part of the complete particle; thin percentage is a lower bound.')
    except WorkflowCancelled:
        print('FindThin cancelled.')
    finally:
        print(f'Images and data: {directory}')
