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

## Index particles in the current FoV

Add `src/scripts/usim_index_particles.py` to Nion Swift's
Scripts panel and run it. It acquires a 512 x 512 HAADF frame using the active
scan profile and creates a new timestamped run directory under
`src/scripts/index_particles_results/` on every execution. Each run saves:

- `index_particles.npz`: original float HAADF data, labels, particle table and metadata.
- `haadf.png`: unannotated grayscale HAADF display image.
- `haadf_indexed.png`: HAADF with green enclosing circles, cyan center crosses,
  and yellow particle IDs matching the array table.

Focus and stage position are preserved. PNGs use the same linear 8-bit display
scaling; use the archive for quantitative intensities. The circles enclose the
detected regions around their centroids and do not measure particle diameter.
An edge particle's circle may extend beyond the image. Output folders are
ignored by Git. The script prints the full run directory after saving.

The algorithm module is `algorithms.IndexParticles`:

```python
from algorithms.IndexParticles import index_particles
result = index_particles(microscope, output_path="particles.npz")
```

For an existing image, use `index_particles_in_image(image, fov_m=...)` and
`result.save("particles.npz")`. Use `result.save_images("output_directory")`
to export the two PNG images. `IndexParticlesConfig` controls Gaussian
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
`center_y_px`, `offset_x_m`, `offset_y_m`, `area_px`, and `touches_edge`.
IDs start at 1, sorted by center row then column, and correspond to the
integer region labels in `labels`; 0 means background. IDs are local to each
acquisition. An empty detection produces an empty table and a zero label map.
Pixel coordinates use x=column and y=row, with integer pixel centers.
Offsets use image axes (x right, y down), relative to the image center.
They are not stage coordinates and do not account for scan rotation.
The scalar FoV denotes the longest image side, assuming square pixels.
The archive also stores schema version 1, image shape, FoV and pixel size in
meters, threshold and detection settings, and a coordinate-system description.
