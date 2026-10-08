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
`find_particles`, `zoom_to_fit`, and `autofocus`; it uses 512 x 512 overview/focus images and 1024 x 1024 final captures.
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

## Find particle edges and acquisition boxes

`algorithms/FindEdge.py` provides `find_edge(microscope, box_fov_m)` and
`find_edge_in_image(image, fov_m=..., box_fov_m=..., particle_id=None)`.
It uses FindParticles segmentation, traces ordered pixel-cell outlines and
covers the selected boundaries with edge-centered squares by default (or a shared square grid). Grid phases are
searched to reduce box count; in the optional grid mode distinct boxes have no area overlap. This is a
heuristic covering, not a globally optimal minimum-box solution.

`result.box_centers_m` is an N x 2 array of image-axis (x, y) offsets in meters.
`result.boxes` contains IDs, pixel centers, meter offsets, box FoV and a flag
for boxes extending outside the image. Use these centers with Zoom2Fit from
the same reference view. `result.save("edge.npz")` records original data,
ordered boundary points (pixels and meter offsets), contour offsets and
particle IDs, and boxes; `result.save_overlay("edge.png")` draws cyan edges,
yellow acquisition squares and red centers. Particle IDs refer to detection
in the supplied image, rather than earlier overview IDs. Holes are filled by
default; set outer_edges_only=False to include internal boundaries.
Frame-touching contours include the image cut and are flagged as incomplete
physical outlines. Boxes smaller than one image pixel are rejected.

Run `python tests/preview_find_edge.py` in nionswift-dev to generate a 10 x 10 nm
box preview using the latest saved particle #1 close-up. Outputs are under
`tests/find_edge_results/<run>/`. Use `--source` and `--output` to override.

## Particle #1 edge autofocus workflow (50 nm boxes)

Add `src/scripts/usim_particle_edge_workflow.py` to Nion Swift's Scripts panel.
Start at the overview containing your intended particle #1. It detects the
current view, centers/fits overview particle #1 with 10% margin, and acquires
a close-up. It selects the region containing the close-up image center (or
the nearest centroid), runs FindEdge with 50 x 50 nm boxes, and for every box
runs Zoom2Fit with no extra margin, autofocus, and a separate HAADF capture.
Overview and focus images are 512 x 512; final per-box photographs are 1024 x 1024. The script needs no file from a previous run.

Results: `src/scripts/particle_edge_workflow_results/<timestamp>/`:

- `overview/`: detected particles and the annotated overview image.
- `particle_0001/`: zoom/reference metadata, `find_edge.npz`, and
  `haadf_edge_boxes.png` for visual checking.
- `particle_0001/edge_box_0001/`, etc.: `haadf.png`, `haadf.npz` (particle ID 1,
  actual FoV, stage, defocus and float image), autofocus arrays/JSON,
  zoom metadata and `box.json` (box ID and center).
- `run.json`: every box's outcome and restoration state.

Each box is anchored to the same particle close-up baseline and reached
directly from the preceding box, preventing accumulated movement. On completion or failure the ORIGINAL overview stage,
FoV and defocus are restored. Individual box errors are logged and processing
continues; nonconverged autofocus captures best measured focus and is flagged.
Do not change the sample, rotation, profile or other microscope controls
while running. Particle #1 uses FindParticles overview numbering, not an
arbitrary specimen ID. Frame-touching edges are flagged in saved metadata.
Customize `ParticleEdgeWorkflowConfig` for other box sizes or focus parameters.

## Edge centering, focus stability, and stopping a workflow

FindEdge now defaults to centers on the traced edge. Covering curved edges
with edge-centered squares can require some overlap; coverage is maintained
and overlap is penalized in the greedy selection. Set `center_on_edge=False`
for the original zero-overlap grid, whose centers need not lie on the edge.
Autofocus averages three frames by median, uses mean-intensity normalization,
limits range expansion, and never expands outside an already bracketed peak.
Flat scores stop with `score_plateau`; exhausted expansion stops with
`expansion_limit`. These are nonconverged results saved explicitly.
The edge workflow starts with a smaller +/- half-FoV focus search.

Click Nion Swift's Cancel to request a cooperative stop. Acquisition waits
poll cancellation and attempt to abort the active record. The current run
report becomes `cancelled`, existing results are retained, and stage/FoV/focus
restoration is attempted with cancellation suspended. Alternatively create an
empty file named `STOP` inside the timestamped run directory printed by the
script. This requests the same stop without using the Scripts dialog.
Startup and device-driver shutdown can still have device-dependent delays;
these changes have automated cancellation tests but need live GUI verification.

The edge workflow now uses a 5 nm minimum focus precision, 5 search positions
per round, and one frame per position. First-box search starts at +/- 4 FoV;
after a converged box, the next starts at that best focus with +/- half FoV.
Focus acquisition uses 512 x 512 pixels; final captures use
1024 x 1024. Overview and edge detection remain 512 x 512. Acquisition settings are restored even on failure or cancellation.
A plateau near a good focus previously counted as nonconvergence when the
requested 0.5 nm precision was finer than the score could resolve.

Edge boxes are now numbered along each traced contour's arclength, starting
near the contour's top-left and traversing its ordered outline (clockwise for
outer boundaries in image coordinates). The table stores `contour_id` and
`arc_length_px`; FindEdge archive schema is now version 2. The overlay shows
box IDs. Each contour is processed consecutively; separate contours are
separate paths. The edge workflow moves directly between absolute targets
computed from the saved particle reference. It no longer returns the stage
to the particle center between boxes. The original overview is restored only
at the end. This reduces unnecessary direction reversals but does not calibrate
or compensate mechanical stage hysteresis.
