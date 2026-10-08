"""Generate a 10 x 10 nm edge-box inspection image for saved particle #1."""
from pathlib import Path
import argparse
import numpy as np
from algorithms.FindEdge import find_edge_in_image
from algorithms.FindParticles import find_particles_in_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidates = sorted((root / 'src/scripts/particle_workflow_results').glob('*/particle_0001/haadf.npz'))
    source = args.source or (candidates[-1] if candidates else None)
    if source is None:
        parser.error('No saved particle_0001/haadf.npz; specify --source')
    output = args.output or root / 'tests/find_edge_results' / source.parent.parent.name
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        image, fov_m = data['image'], float(data['fov_m'])
    detected = find_particles_in_image(image, fov_m=fov_m)
    if not len(detected.particles):
        raise RuntimeError('No particle found in the saved particle #1 image')
    # The original workflow ID differs from IDs re-detected in the close-up.
    distances = detected.particles['offset_x_m']**2 + detected.particles['offset_y_m']**2
    particle_id = int(detected.particles[np.argmin(distances)]['id'])
    result = find_edge_in_image(image, fov_m=fov_m, box_fov_m=10e-9, particle_id=particle_id)
    result.save(output / 'particle_0001_edge_10nm.npz')
    path = result.save_overlay(output / 'particle_0001_edge_10nm.png')
    (output / 'source.txt').write_text(f'Source: {source.resolve()}\nFoV: {fov_m * 1e9:.6f} nm\nBox: 10 x 10 nm\nBoxes: {len(result.boxes)}\n', encoding='utf-8')
    print(f'{len(result.boxes)} boxes. Preview: {path.resolve()}')


if __name__ == '__main__':
    main()
