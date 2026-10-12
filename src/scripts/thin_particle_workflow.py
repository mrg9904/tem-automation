"""Map particles from a fixed overview and checkpoint a descending thin ranking."""
import csv
from dataclasses import asdict
from pathlib import Path
import numpy as np
from adapters.cancellation import check_cancelled, WorkflowCancelled
from algorithms.Zoom2Fit import zoom_to_fit
from algorithms.autofocus import autofocus, AutofocusConfig
from algorithms.FindThin import find_thin, FindThinConfig, save_top_thin_regions
from algorithms.AutoBC import auto_bc
from scripts.common import write_json


AUTOFOCUS_CONFIG = AutofocusConfig.coarse(precision_m=100e-9)


def save_ranking(directory, report):
    """Keep successful results ranked even if a later acquisition is stopped."""
    ranked = sorted((entry for entry in report['particles'] if entry.get('thin_area_percent') is not None),
                    key=lambda entry: (-entry['thin_area_percent'], entry['particle_id']))
    report['ranking'] = [dict(rank=index, particle_id=entry['particle_id'],
        thin_area_percent=entry['thin_area_percent'], directory=entry['directory'],
        autofocus_converged=entry['autofocus']['converged'],
        unquantified_area_percent=entry['unquantified_area_percent'],
        area_outside_current_fov_percent=entry['area_outside_current_fov_percent'])
        for index, entry in enumerate(ranked, 1)]
    write_json(directory/'thin_workflow.json', report)
    write_json(directory/'thin_ranking.json', report['ranking'])
    columns = ['rank', 'particle_id', 'thin_area_percent', 'directory', 'autofocus_converged',
               'unquantified_area_percent', 'area_outside_current_fov_percent']
    temporary = directory/'thin_ranking.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report['ranking'])
    temporary.replace(directory/'thin_ranking.csv')
    np.savez_compressed(directory/'thin_ranking.npz',
        particle_id=np.asarray([entry['particle_id'] for entry in ranked], dtype=np.int64),
        thin_area_percent=np.asarray([entry['thin_area_percent'] for entry in ranked], dtype=float),
        rank=np.arange(1, len(ranked)+1, dtype=np.int64))


def map_particles(microscope, particles, center_statuses, reference_stage_xy_m, directory, *,
                  window_report, config=FindThinConfig(), autofocus_config=AUTOFOCUS_CONFIG,
                  defocus_offset_m=0., progress=print):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(particles) != len(center_statuses):
        raise ValueError('Center statuses must match the particle array')
    autofocus_config.validate()
    if not np.isfinite(defocus_offset_m):
        raise ValueError('Defocus offset must be finite')
    report = dict(window=window_report, status='running', particle_count=len(particles), particles=[],
                  mapping_defocus_mode='relative_to_autofocus',
                  mapping_defocus_offset_m=float(defocus_offset_m), autofocus_image_size_px=512,
                  mapping_config=asdict(config), autofocus_config=asdict(autofocus_config))
    focus_seed = float(microscope.get_defocus())
    save_ranking(directory, report)
    try:
        for particle, center_status in zip(particles, center_statuses):
            check_cancelled()
            identifier = int(particle['id'])
            target_directory = directory/'particles'/f'particle_{identifier:04d}'
            target_directory.mkdir(parents=True, exist_ok=True)
            entry = dict(particle_id=identifier, directory=str(target_directory.relative_to(directory)),
                         center_status=str(center_status), status='running')
            report['particles'].append(entry)
            save_ranking(directory, report)
            prefix = f"Particle #{identifier} ({len(report['particles'])}/{len(particles)})"
            if center_status not in ('complete', 'recovered'):
                entry.update(status='skipped_unresolved_center', reason='No unique complete center in the overview context')
                progress(f'{prefix}: center unresolved; skipping and recording this result.')
                write_json(target_directory/'particle.json', entry)
                save_ranking(directory, report)
                continue
            try:
                progress(f'{prefix}: moving and zooming to the particle...')
                zoom = zoom_to_fit(microscope,
                    (float(particle['offset_x_m']), float(particle['offset_y_m'])),
                    float(particle['circle_diameter_m']), padding=1.1,
                    reference_stage_position_m=tuple(reference_stage_xy_m))
                entry['zoom'] = asdict(zoom)
                # Seed from the preceding measured focus, not a mapping offset.
                microscope.set_defocus(focus_seed)
                progress(f'{prefix}: running {autofocus_config.mode} autofocus at 512x512 '
                         f'(target precision {autofocus_config.coarse_precision_m*1e9:g} nm)...'
                         if autofocus_config.mode == 'coarse' else f'{prefix}: running fine autofocus at 512x512...')
                with microscope.acquisition_settings(image_size_px=512, dwell_time_us=1.):
                    focus = autofocus(microscope, config=autofocus_config,
                                      progress=lambda message: progress(f'{prefix}: {message}'))
                entry['autofocus'] = asdict(focus)
                write_json(target_directory/'autofocus.json', entry['autofocus'])
                if focus.converged:
                    focus_seed = focus.best_defocus_m
                progress(f'{prefix}: autofocus {"converged" if focus.converged else "did not converge; using best measured focus"}, '
                         f'focus = {focus.best_defocus_m*1e9:.1f} nm.')
                result = find_thin(microscope, output_directory=target_directory/'thickness', config=config,
                    focus_reference_m=focus.best_defocus_m, defocus_offset_m=defocus_offset_m,
                    progress=lambda message: progress(f'{prefix}: {message}'))
                summary = result.summary()
                entry.update(status='mapped' if focus.converged else 'mapped_unconverged',
                    thin_area_percent=summary['thin_area_percent'],
                    unquantified_area_percent=summary['unquantified_area_percent'],
                    area_outside_current_fov_percent=summary['area_outside_current_fov_percent'],
                    mapping_lens_defocus_m=float(focus.best_defocus_m+defocus_offset_m))
                microscope.persist_last_scan_profile()
                tone = auto_bc(result.haadf, result.particle_mask)
                microscope.set_display_tone(tone.brightness, tone.contrast)
                progress(f"{prefix}: mapping complete, thin area = {summary['thin_area_percent']:.2f}%.")
            except WorkflowCancelled:
                entry['status'] = 'cancelled'
                write_json(target_directory/'particle.json', entry)
                raise
            except Exception as error:
                entry.update(status='failed', error=str(error))
                progress(f'{prefix}: failed: {error}. Continuing with the next particle.')
            write_json(target_directory/'particle.json', entry)
            save_ranking(directory, report)
        report['status'] = ('complete' if all(entry['status'] == 'mapped' for entry in report['particles'])
                            else 'complete_with_issues')
    except WorkflowCancelled:
        report['status'] = 'cancelled'
        raise
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        save_ranking(directory, report)
    progress(f"Mapping finished: {len(report['ranking'])}/{len(particles)} particles mapped. "
             'Saving thin-area ranking from highest to lowest...')
    progress('Selecting and marking broad thin regions in the top three particles...')
    report['top_thin_regions'] = save_top_thin_regions(directory, report['ranking'])
    save_ranking(directory, report)
    return report
