# TEM Automation

The runtime code is deliberately divided into only three layers:

```text
src/
├── adapters/
│   ├── base.py
│   └── nion_usim.py
├── algorithms/
│   └── autofocus.py
└── scripts/
    └── usim_haadf_autofocus.py
```

- `adapters`: defines a small vendor-independent microscope API, stores the
  current acquisition configuration, and translates calls to a vendor API.
- `algorithms`: owns algorithm-specific defaults and calls only the
  vendor-independent microscope API.
- `scripts`: initializes an instrument and lists the high-level experiment
  actions to execute.

The `tests` directory is development support and is not part of the runtime
architecture.

## Install in the Nion Swift development environment

```bat
conda activate nionswift-dev
cd /d D:\Development\tem-automation
python -m pip install -e .
```

## Test

```bat
python -m unittest discover -s tests -v
```

## Run in Nion Swift

Start Nion Swift and uSim, then choose `File > Scripts...` and add:

```text
D:\Development\tem-automation\src\scripts\usim_haadf_autofocus.py
```

Double-click the script to run HAADF autofocus. The script contains only the
instrument initialization and the high-level `autofocus(microscope)` action.

Autofocus initially searches the current defocus +/- one active profile FoV
(a total width of twice the FoV). If the highest focus score lies at either
endpoint, it centers on that endpoint and doubles the half-range until a
peak is bracketed. It then refines to 1% of the FoV. The default 12-round
budget bounds both expansion and refinement; an unbracketed search reports
`converged=False` and retains the best measured focus. Acquisition failures
restore the original defocus.

## Find particles in the current FoV

Add `src/scripts/usim_find_particles.py` to Nion Swift's
Scripts panel and run it. It acquires a 512 x 512 HAADF frame using the active
scan profile and creates a new timestamped run directory under
`src/scripts/find_particles_results/` on every execution. Each run saves:

- `find_particles.npz`: original float HAADF data, labels, particle table and metadata.
- `haadf.png`: unannotated grayscale HAADF display image.
- `haadf_annotated.png`: HAADF with green enclosing circles, cyan center crosses,
  and yellow particle IDs matching the array table.

Focus and stage position are preserved. PNGs use the same linear 8-bit display
scaling; use the archive for quantitative intensities. The circles enclose the
detected regions around their centroids and do not measure particle diameter.
An edge particle's circle may extend beyond the image. Output folders are
ignored by Git. The script prints the full run directory after saving.

The algorithm module is `algorithms.FindParticles`:

```python
from algorithms.FindParticles import find_particles
result = find_particles(microscope, output_path="particles.npz")
```

For an existing image, use `find_particles_in_image(image, fov_m=...)` and
`result.save("particles.npz")`. Use `result.save_images("output_directory")`
to export the two PNG images. `FindParticlesConfig` controls Gaussian
smoothing, the minimum region area (16 pixels by default), a manual threshold
(default: Otsu), bright/dark particles, and exclusion of edge particles.
The initial implementation detects separated bright regions. Touching or
projected overlapping particles form one region; dim particles and strongly
varying backgrounds may need a manual threshold. Edge particles are included
and flagged by default; their detected center is the visible region's center.

```python
import numpy as np
with np.load("particles.npz", allow_pickle=False) as data:
    particles = data["particles"]
    labels = data["labels"]
    image = data["image"]
```

`particles` is a structured array with fields `id`, `center_x_px`,
`center_y_px`, `offset_x_m`, `offset_y_m`, `area_px`, `circle_diameter_px`,
`circle_diameter_m`, and `touches_edge`. Circle diameters are saved in pixels
and meters and are used directly to draw the PNG overlay. The diameter is
`2 * max(4, maximum pixel-center distance from centroid + 1)` pixels; it is
an annotation extent, not a measured particle diameter.
IDs start at 1, sorted by center row then column, and correspond to the
integer region labels in `labels`; 0 means background. IDs are local to each
acquisition. An empty detection produces an empty table and a zero label map.
Pixel coordinates use x=column and y=row, with integer pixel centers.
Offsets use image axes (x right, y down), relative to the image center.
They are not stage coordinates and do not account for scan rotation.
The scalar FoV denotes the longest image side, assuming square pixels.
The archive also stores schema version 2 (adds the circle diameter fields), image shape, FoV and pixel size in
meters, threshold and detection settings, and a coordinate-system description.

## Center and zoom to a particle

`algorithms/Zoom2Fit.py` exports `zoom_to_fit`. Center coordinates are `(x, y)`
in meters relative to the current image center; diameter is in meters.

```python
from algorithms.Zoom2Fit import zoom_to_fit
particle = result.particles[0]  # From FindParticles for the current view.
zoom = zoom_to_fit(
    microscope,
    (particle["offset_x_m"], particle["offset_y_m"]),
    particle["circle_diameter_m"],
    padding=1.1,  # Optional 10% margin; default is 1.0 (FoV = diameter).
)
print(zoom.actual_fov_m)
```

The uSim adapter accounts for scan rotation and moves the stage using the same
convention as interactive double-click centering. FoV is saved to the active
profile while preserving other profile settings. The result records requested
and actual FoV and before/after stage positions. Defocus is unchanged; this
action does not acquire a new image. Failures attempt to restore both FoV and
stage position, and report incomplete restoration explicitly.
Only use offsets from the current view: saved offsets become stale after
moving the stage, changing scan center or rotation. Recenter or re-detect before
selecting another particle. A scalar size assumes a square acquisition FoV.

## Find particles, autofocus, and capture every particle

Add `src/scripts/usim_particle_workflow.py` to Nion Swift's Scripts panel.
Select the overview sample/FoV and run it. The modular workflow composes
`find_particles`, `zoom_to_fit`, and `autofocus`; it acquires 512 x 512 images.
No initial overview autofocus is performed: detection uses the current focus.

Each run creates `src/scripts/particle_workflow_results/<timestamp>/`:

- `overview/`: particle table, original HAADF, and circles/centers/IDs.
- `particle_0001/`, etc.: `haadf.npz` (float image, stage x/y in meters,
  actual FoV, pixel size and defocus), `haadf.png`, `zoom.json`,
  `autofocus.json`, `autofocus.npz` (focus positions/scores), and `particle.json`.
- `run.json`: initial state, configuration, particle count, per-particle
  outcomes, and whether the original instrument state was restored.

Before each particle, the workflow restores the overview stage/FoV/defocus,
then centers on the saved offset and zooms to its circle diameter plus 10%
margin. It autofocuses and acquires a separate final HAADF frame.
At completion or failure, it attempts to restore the original stage/FoV/defocus.
Individual errors are logged and processing continues by default; restoration
failures abort the run. Nonconverged autofocus still records the best measured
focus and marks the capture `captured_unconverged` explicitly. An empty overview
saves the overview and run report without taking per-particle captures.
Do not change the sample, selected profile, scan rotation, beam shift, or
stage controls while the workflow runs. Overview detection limitations still
apply: touching particles can be merged, and edge regions have partial centers.

For custom settings or a different output path:

```python
from scripts.usim_particle_workflow import ParticleWorkflowConfig, run_particle_workflow
from algorithms.autofocus import AutofocusConfig
config = ParticleWorkflowConfig(padding=1.2, continue_on_error=False,
                               autofocus=AutofocusConfig(max_rounds=12))
run_particle_workflow(microscope, "output_directory", config=config)
```
