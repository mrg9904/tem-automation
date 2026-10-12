"""Nion Swift: find window, subdivide, zoom to cell #1 and then particle #1."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from time import monotonic
import numpy as np
import json
from dataclasses import asdict
from PIL import Image, ImageDraw
from scripts import __file__ as scripts_package_file
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from adapters.nion_usim import nion_cancel_callback
from adapters.cancellation import cancellation_scope, WorkflowCancelled
from algorithms.FindWindow import find_window_in_image, make_window_anchor, subdivide_window
from algorithms.FindParticles import find_particles
from algorithms.Zoom2Fit import zoom_to_fit
from algorithms.AutoBC import auto_bc


def run_window_workflow(microscope, directory, *, progress=print, zoom_first_particle=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    progress('Initializing sample and stage...')
    microscope.initialize_window_view()
    progress('Acquiring Ronchigram and finding the central window...')
    acquisition_started = monotonic()
    image = microscope.acquire_ronchigram()
    ronchigram_acquisition_seconds = monotonic()-acquisition_started
    progress(f'Ronchigram acquired in {ronchigram_acquisition_seconds:.1f} s.')
    np.savez_compressed(directory/'capture.npz', image=image,
        stage_xy_m=microscope.get_stage_position(), stage_z_m=600e-6)
    result = find_window_in_image(image)
    anchor = make_window_anchor(result, *microscope.window_coordinates(
        np.vstack((result.center_xy_px, result.corners_xy_px))))
    result.save(directory, anchor=anchor)
    progress('Window found. Subdividing into a 5x5 grid and saving labels...')
    grid = subdivide_window(result, anchor)
    grid.save(result, directory)
    cell = grid.cells[0]
    progress('Returning Stage Z to 0 for HAADF scanning...')
    microscope.prepare_window_scan(cell['scan_rotation_rad'])
    reference = microscope.get_stage_position()
    offset = microscope.get_coordinate_transform().image_offset_for_stage_position(
        cell['center_stage_xy_m'], reference)
    progress('Moving and zooming to cell #1...')
    cell_zoom = zoom_to_fit(microscope, tuple(offset), max(cell['size_m']),
                           reference_stage_position_m=reference)
    cell_directory = directory/'cell_0001'
    cell_directory.mkdir(exist_ok=True)
    with microscope.acquisition_settings(image_size_px=1024):
        progress('Acquiring cell HAADF, finding particles and checking clipped centers in a wider view...')
        particles = find_particles(microscope, output_path=cell_directory/'find_particles.npz')
        particles.save_images(cell_directory)
        report = dict(cell=cell, cell_zoom=asdict(cell_zoom), particle_count=len(particles.particles),
                      ronchigram_acquisition_seconds=ronchigram_acquisition_seconds,
                      cell_reference_stage_xy_m=list(microscope.get_stage_position()),
                      center_statuses=particles.center_statuses.tolist())
        if len(particles.particles) and zoom_first_particle:
            if particles.center_statuses[0] not in ('complete', 'recovered'):
                report['status'] = 'unresolved_particle_center'
                (directory/'workflow.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                raise RuntimeError('Particle #1 center is unresolved; inspect the saved context images before moving')
            progress(f'Found {len(particles.particles)} particles. Moving to particle #1...')
            particle = particles.particles[0]
            particle_zoom = zoom_to_fit(microscope,
                (float(particle['offset_x_m']), float(particle['offset_y_m'])),
                float(particle['circle_diameter_m']))
            progress('Acquiring particle #1 HAADF and adjusting brightness/contrast...')
            target = microscope.acquire_haadf()
            target_directory = cell_directory/'particle_0001'
            target_directory.mkdir(exist_ok=True)
            np.savez_compressed(target_directory/'haadf.npz', image=target,
                stage_xy_m=microscope.get_stage_position(), fov_m=microscope.get_fov())
            tone = auto_bc(target)
            picture = Image.fromarray(tone.render(target))
            report['auto_bc'] = tone.metadata()
            picture.save(target_directory/'haadf.png')
            labeled = picture.convert('RGB')
            draw = ImageDraw.Draw(labeled)
            x, y = (target.shape[1]-1)/2, (target.shape[0]-1)/2
            draw.line((x-10, y, x+10, y), fill='cyan', width=2)
            draw.line((x, y-10, x, y+10), fill='cyan', width=2)
            draw.text((10, 10), 'Cell #1 / target particle #1', fill='yellow', stroke_width=1, stroke_fill='black')
            labeled.save(target_directory/'haadf_target.png')
            report['particle_zoom'] = asdict(particle_zoom)
        microscope.persist_last_scan_profile()
        if len(particles.particles) and zoom_first_particle:
            microscope.set_display_tone(tone.brightness, tone.contrast)
    (directory/'workflow.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    progress('Window workflow complete.')
    return report


def script_main(api_broker):
    directory = Path(scripts_package_file).resolve().parent/'find_window_results'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with cancellation_scope(nion_cancel_callback(print)):
            with NionUSimRonchigramAdapter(api_broker.get_api(version='~1.0'), timeout_s=120.) as microscope:
                report = run_window_workflow(microscope, directory)
        print(f"Cell #1: found {report['particle_count']} particles. " +
              ('Centered on particle #1.' if report['particle_count'] else 'No particle to zoom to.'))
    except WorkflowCancelled:
        print('FindWindow cancelled.')
    except Exception as error:
        print(f'***Warning*** {error}')
    finally:
        print(f'Images and data: {directory}')
