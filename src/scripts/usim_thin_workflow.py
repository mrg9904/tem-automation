"""Nion Swift entry: window -> cell #1 -> all particles -> ranked thickness maps."""
from datetime import datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4
import numpy as np
from scripts import __file__ as scripts_package_file
from scripts.window_particle_workflow import run_window_workflow
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from adapters.nion_usim_eels import NionUSimEELSAdapter
from adapters.nion_usim import nion_cancel_callback
from adapters.cancellation import cancellation_scope, WorkflowCancelled, check_cancelled
from algorithms.FindThin import FindThinConfig
from scripts.thin_particle_workflow import map_particles, save_ranking, AUTOFOCUS_CONFIG

CONFIG = FindThinConfig(image_size_px=128, pixel_time_us=10., dispersion_eV=2., energy_max_eV=498.)


def run_thin_workflow(api, directory, *, config=CONFIG, autofocus_config=AUTOFOCUS_CONFIG,
                      defocus_offset_m=0., progress=print):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = monotonic()

    def notify(message):
        progress(f'[{monotonic()-started:.1f}s] {message}')

    check_cancelled()
    with NionUSimRonchigramAdapter(api, timeout_s=120.) as microscope:
        window = run_window_workflow(microscope, directory/'window', progress=notify, zoom_first_particle=False)
    if not window['particle_count']:
        raise RuntimeError('No particles found in cell #1; stopping thickness mapping.')
    check_cancelled()
    with np.load(directory/'window/cell_0001/find_particles.npz', allow_pickle=False) as saved:
        particles = saved['particles'].copy()
        center_statuses = saved['center_status'].copy()
    with NionUSimEELSAdapter(api, timeout_s=120.) as microscope:
        report = map_particles(microscope, particles, center_statuses,
            window['cell_reference_stage_xy_m'], directory, window_report=window,
            config=config, autofocus_config=autofocus_config, defocus_offset_m=defocus_offset_m, progress=notify)
    report['elapsed_seconds'] = monotonic()-started
    save_ranking(directory, report)
    notify('Thin workflow complete.')
    return report


def script_main(api_broker):
    directory = Path(scripts_package_file).resolve().parent/'thin_workflow_results'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8])
    try:
        with cancellation_scope(nion_cancel_callback(print)):
            run_thin_workflow(api_broker.get_api(version='~1.0'), directory)
    except WorkflowCancelled:
        print('Thin workflow cancelled.')
    except Exception as error:
        print(f'***Warning*** Thin workflow failed: {error}')
    finally:
        print(f'Images and data: {directory}')
