"""Reanalyze the same saved cell frame; display stretching never enters detection."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from algorithms.FindParticles import FindParticlesConfig, find_particles_in_image


def main():
    cell = Path(__file__).resolve().parent/'find_window_results/preview/cell_0001'
    with np.load(cell/'find_particles.npz', allow_pickle=False) as captured:
        image, fov = captured['image'].copy(), float(captured['fov_m'])
    legacy = find_particles_in_image(image, fov_m=fov,
        config=FindParticlesConfig(threshold_method='otsu'))
    result = find_particles_in_image(image, fov_m=fov)
    output = cell/'thin_particle_review'
    output.mkdir(exist_ok=True)
    result.save(output/'find_particles.npz')
    result.save_images(output)
    lo, hi = np.percentile(image, [1, 99.5])
    display = np.arcsinh(np.maximum(image-lo, 0)/max((hi-lo)*.01, 1e-12))
    replace(result, image=display).save_images(output/'display_asinh')
    report = dict(otsu_count=len(legacy.particles), local_snr_count=len(result.particles),
        display='asinh display only; detection and saved NPZ use the original linear data')
    (output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report)
    print(output)


if __name__ == '__main__':
    main()
