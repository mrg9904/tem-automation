"""Recheck saved thin mappings without acquiring or changing original data."""
import copy
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from algorithms.FindThin import FindThinConfig, analyze_thin, complete_particle_footprint
from scripts.thin_particle_workflow import save_ranking
from scripts.common import write_json


def main():
    repository = Path(__file__).resolve().parents[1]
    runs = repository/'src/scripts/thin_workflow_results'
    source = max((directory for directory in runs.iterdir() if (directory/'thin_workflow.json').exists()),
                 key=lambda directory: directory.stat().st_mtime)
    output = repository/'tests/find_thin_results'/('area_review_'+source.name)
    output.mkdir(parents=True, exist_ok=True)
    report = copy.deepcopy(json.loads((source/'thin_workflow.json').read_text()))
    report['area_review_source_directory'] = str(source)
    for entry in report['particles']:
        if entry.get('thin_area_percent') is None:
            continue
        directory = source/entry['directory']/'thickness'
        with np.load(directory/'particle_context.npz', allow_pickle=False) as context, \
             np.load(directory/'find_thin.npz', allow_pickle=False) as saved:
            labels, particles = context['labels'], context['particles']
            chosen = min(particles, key=lambda particle: np.hypot(particle['offset_x_m'], particle['offset_y_m']))
            complete = complete_particle_footprint(labels, chosen['id'])
            fov, context_fov = float(saved['fov_m']), float(context['fov_m'])
            shape = saved['particle_mask'].shape
            pixel, context_pixel = fov/max(shape), context_fov/max(labels.shape)
            yy, xx = np.mgrid[:shape[0], :shape[1]]
            mask = ndimage.map_coordinates(complete.astype(np.uint8),
                [(yy+.5-shape[0]/2)*pixel/context_pixel+labels.shape[0]/2-.5,
                 (xx+.5-shape[1]/2)*pixel/context_pixel+labels.shape[1]/2-.5],
                order=0, mode='constant', cval=0) > 0
            previous = json.loads(str(saved['metadata_json']))
            metadata = dict(previous['acquisition'])
            if not mask.any():
                with np.load(directory/'particle_footprints.npz', allow_pickle=False) as fallback:
                    labels, particles = fallback['labels'], fallback['particles']
                    chosen = min(particles, key=lambda particle: np.hypot(particle['offset_x_m'], particle['offset_y_m']))
                    complete = complete_particle_footprint(labels, chosen['id'])
                    mask = complete
                    context_pixel = pixel
            if not chosen['touches_edge']:
                metadata['particle_full_area_m2'] = float(complete.sum()*context_pixel**2)
            metadata.update(particle_context_label_id=int(chosen['id']),
                particle_context_included_label_ids=np.unique(labels[complete]).tolist(),
                particle_footprint_method='selected region plus enclosed segmented foreground; background holes preserved',
                area_review_source_directory=str(directory),
                spectrum_image_file=str(directory/'eels_spectrum_image.npy'))
            cube = np.load(directory/'eels_spectrum_image.npy', mmap_mode='r')
            result = analyze_thin(cube, saved['energy_eV'], particle_mask=mask,
                fov_m=fov, haadf=saved['haadf'], config=FindThinConfig(**previous['config']), metadata=metadata)
            target = output/entry['directory']
            result.save(target/'thickness')
            summary = result.summary()
            entry['original_thin_area_percent'] = entry['thin_area_percent']
            for key in ['thin_area_percent', 'unquantified_area_percent', 'area_outside_current_fov_percent']:
                entry[key] = summary[key]
            write_json(target/'particle.json', entry)
            if entry['particle_id'] in (21, 25):
                print(f"Particle #{entry['particle_id']}: {entry['original_thin_area_percent']:.2f}% -> "
                      f"{entry['thin_area_percent']:.2f}%, thin {summary['thin_area_px']}/{summary['particle_area_px']} pixels")
    save_ranking(output, report)
    print(output)


if __name__ == '__main__':
    main()
