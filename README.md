# TEM Automation

TEM automation with three runtime layers: device adapters, reusable algorithms,
and experiment scripts. Nion Swift/uSim is the current device backend.

```text
src/
  adapters/
    base.py                 device protocols and capability validation
    coordinates.py          shared image/stage transforms
    cancellation.py         generic cooperative cancellation
    nion_usim.py             Nion API, acquisition, profiles, UI cancellation binding
  algorithms/
    FindParticles.py        particle detection and indexing
    FindEdge.py             ordered contour tracing and edge boxes
    FindWindow.py           square grid opening geometry and scan anchors
    Zoom2Fit.py             positioning and FoV fitting
    autofocus.py            focus search and configuration validation
    FindROI.py              stable public ROI API
    roi/                    features, reference, detection, results, output, mosaic
  scripts/
    common.py               typed capture pipeline, checkpoints, state and logging
    usim_particle_workflow.py
    usim_particle_edge_workflow.py
    usim_find_particles.py
    usim_find_roi.py
    usim_haadf_autofocus.py
```

Adapters do not import algorithms or scripts. Algorithms do not import scripts
or the Nion backend. Scripts choose experiment policy and orchestrate public
operations. Workflows share helpers through `scripts.common`, rather than
importing private helpers from another experiment script.
See [architecture and contracts](docs/architecture.md).

## Install and test

```bat
conda activate nionswift-dev
cd /d D:\Development\tem-automation
python -m pip install -e . --no-deps --no-build-isolation
python -m unittest discover -s tests -v
```

Dependencies: Python >=3.10, NumPy >=1.24, SciPy >=1.10 and Pillow >=10.1.
There is no PyTorch or pretrained-model download requirement.

## Find thin particle areas with EELS

Import `src/scripts/usim_find_thin.py` after positioning the particle. It keeps
the current stage/FoV, sets effective defocus to +FoV/10 (compensating Stage Z),
and acquires a 128x128 EELS spectrum image at 10 us per pixel. `CONFIG` exposes
scan size, dwell, dispersion and spectral bounds. Defaults are 2 eV/channel
and -10..498 eV, 255 channels; the float32 cube is about 17 MB. For a wider
window use `FindThinConfig(dispersion_eV=4., energy_max_eV=4082.)`.

`algorithms.FindThin` computes `t/lambda = ln(It/I0)` solely from the measured
count spectra. It integrates all recorded channels for It and channels within
0 +/-5 eV for I0. It does not subtract a power-law scattering background or
read the simulator's thickness/tau. Pixels inside the particle footprint with
valid counts and t/lambda <1 are marked thin. Invalid/zero-ZLP pixels are
unquantified, rather than thin. Percentage is thin area divided by the entire
segmented particle footprint, including any unquantified pixels in the denominator.
Particle masks can be supplied explicitly. Otherwise an additional concentric
2x HAADF overview is used to segment the central particle with adequate
background, then project its footprint onto SI pixels. This helps retain thin
edges that a close-up's thick core would suppress. A complete overview footprint
also supplies the full-particle area denominator; any part not covered by EELS
counts as unquantified and the measured thin percentage is a lower bound.
If context segmentation is unavailable, the acquired HAADF is used with an
Otsu fallback. Masks remain image-derived estimates and can miss weak boundaries.
Without a complete context footprint, a clipped particle is explicitly reported
as visible-area percentage, not an estimate of its full area outside the FoV.

Results are in `src/scripts/find_thin_results/<timestamp>/`:
raw spectrum images are processed in memory and are not saved. Existing saved
cubes are left untouched. `find_thin.npz` holds thickness/masks,
`thin_regions.json` holds area percentages and thin-region positions,
`haadf_thin_annotated.png` shows thin pixels in green, particle boundary in
yellow and unquantified pixels in red. `t_over_lambda.png` is a 0..2 display
(green to red; outside footprint black, invalid gray); NPZ retains actual values.
Detector energy settings are restored after acquisition. Cancellation is checked
between spectrum-bank entries and scan blocks.

uSim requires the new `EELSCameraSimulator.acquire_spectrum_image` backend.
It batches the existing compound-Poisson spectra using optical-depth bins of
0.01 per material and the live camera's shot-noise model. It uses cached
geometry projections and bilinear scan-rotation sampling for cathode specimens,
avoids a 256-row detector image at every probe pixel, and streams eight rows at
a time. A bounded 4096-entry probability cache reuses repeated spectra while
sampling fresh noise for each acquisition. This is a simulator batch path; HAADF is a separate scan at the same
view, and the present EELS model uses a point probe even when defocus is set.
These acquisition limits are recorded in metadata. For actual hardware,
the adapter would need a synchronized detector acquisition implementation.

A <500 eV window saves data and processing, but unrecorded high-energy losses
reduce It and can underestimate t/lambda, especially near the cutoff at 1.
No model-based tail correction is silently applied. The initial full-size
512x512/4 eV-channel preview is under `tests/find_thin_results/preview/`;
`python -m tests.preview_find_thin` generates the faster 128x128 example under
`tests/find_thin_results/preview_128/` at the saved first-particle target.
Thin-region positions are selected inside each actual thin component, rather
than using a centroid that may fall in the thick center of an annular region.

## Find a grid window from the Ronchigram

On branch `findWindow`, import `src/scripts/usim_find_window.py` in Nion Swift.
It selects `1000CathodeParticleOnCarbon`, sets Stage X=938 nm, Y=-6820 nm,
Z=600 um and defocus=0, and parks the probe at the scan center. The updated
uSim startup also uses this sample and its initial position; restart Swift to
apply startup changes. The uSim changes are in its `build_cathode` checkout.

`algorithms.FindWindow.find_window_in_image` uses only the image and a square
shape constraint. It tests both contrast polarities, fills internal holes,
rejects clipped components, and fits a square to the outer contour. It selects
the complete square nearest the detector image center. It does not read STL
metadata, assume a 54 um size, or assume a 60 degree rotation. On a uniform
grid, geometry alone cannot identify a particular grid index: positioning the
desired opening near image center establishes which window is selected.

Results are under `src/scripts/find_window_results/<timestamp>/`:
`ronchigram_window.png` labels the four corners, center, side and angle;
`find_window.npz` preserves image/geometry/JSON metadata, and `window.json`
records detector geometry and calibrated Stage targets. The adapter uses the
native Ronchigram ray mapping, including display rotation and defocus sign,
to convert the fitted points. No scan FoV is used as camera magnification.
The anchor includes a normalized bilinear Stage patch for future subdivisions;
interior positions assume weak ray distortion and are not yet a scanning workflow.

Screen angles are clockwise from x right toward y down. Detector angles use
the signed angular calibrations. Both are modulo 90 degrees because a square
has four equivalent orientations. A complete visible opening and adequate
contrast are required; substantial tilt or aberrations can distort a square
enough to reject it rather than silently infer unsupported geometry.

`python -m tests.preview_find_window` acquires a noisy simulated Ronchigram at
the requested view and saves a local labeled example in `find_window_results/preview`.

Import `src/scripts/usim_window_particle_workflow.py` for the full sequence.
The original `usim_find_window.py` retains window detection only.
The new entry script calls `subdivide_window` for a 5x5 grid, inset 1 um from
each measured edge. A 54 um window gives a 52 um usable square and 10.4 um
cells; measured sizes are retained rather than forcing a nominal 54 um width.
Cells follow the window axes and IDs trace a serpentine path starting at C1.
`ronchigram_grid.png`, `window_grid.json` and `window_grid.npz` retain all
25 cells, their corners, centers and Stage targets. Distorted physical anchors
are rejected rather than assigning incorrect inset distances.

It then aligns scan rotation to the grid and returns Stage Z to zero,
uses Zoom2Fit on cell #1, captures 1024x1024 HAADF and calls FindParticles.
It uses Zoom2Fit again on particle #1 (IDs local to that cell), saves its
image, and retains the final scan profile. No autofocus is performed.
Connected/overlapping particles can be one detection, and edge-clipped
particles are flagged; these are the existing FindParticles limitations.
If the first cell has no detections, the workflow retains that cell's image.

FindParticles now defaults to `threshold_method="local_snr"` to retain thin,
weak-contrast particles beside bright thick ones. It removes a robust fitted
illumination slope, clips strong signals for local background estimation,
and measures background variance with a robust noise floor. Seeds require
5 times that local scale and boundaries grow at 3 times the scale; watershed
preserves separate bright cores during growth. These scores include substrate
variation and are not statistical detection probabilities. This still requires
resolvable contrast; touching particles without distinct seeds can remain joined.
NPZ retains `background` and `noise_scale` maps and the effective method.
Its scalar `threshold` is the strong-core Otsu threshold for local SNR, not
the boundary threshold. Explicit `threshold=` overrides the adaptive method;
`FindParticlesConfig(threshold_method="otsu")` restores the earlier behavior.
Exact constant-background images without a measurable noise scale fall back
to Otsu. Defaults now apply to all callers, including the window workflow.

The acquisition wrapper `find_particles` now recovers edge-particle centers
when the adapter supports `acquire_particle_context`. It captures a concentric
2x overview, optionally 4x if a matched particle remains clipped, then matches
foreground overlap to complete objects. FoV, scan profile and primary-frame
record metadata are restored. Context resolution grows up to 1536 pixels to
bound simulator memory; the actual context pixel size is recorded.
Centers and circle diameters then describe the complete segmented object,
including centers outside the original image. IDs and `area_px` still describe
the original visible detection. NPZ adds `visible_centers_xy_px`, `center_status`
and `full_segmented_area_m2`; original `touches_edge` is retained. Status
`recovered`/`complete` supports whole-object positioning; `clipped`, `unmatched`,
`ambiguous` or `context_clipped` does not claim a full center. The image-only
`find_particles_in_image` cannot recover unseen geometry without context.
Context images and matched original IDs are saved in `find_particles_context_01/`
(and possibly `_02/`), with a yellow rectangle marking the original view.
The window workflow refuses to position particle #1 if its center remains unresolved.
`python -m tests.preview_thin_particles` compares both methods on the identical
saved cell frame and writes `cell_0001/thin_particle_review/`. Its
`display_asinh/haadf_annotated.png` stretches display intensity to expose thin
particles; this nonlinear display transform never enters detection or raw NPZ.

`python -m tests.preview_window_workflow` runs this complete real-uSim pipeline
and saves review images under `tests/find_window_results/preview/`:
`ronchigram_grid.png`, `cell_0001/haadf_annotated.png`, and
`cell_0001/particle_0001/haadf_target.png`. `workflow.json` records both zooms.

## Run the edge experiment

In Nion Swift, open `File > Scripts...` and add
`src/scripts/usim_particle_edge_workflow.py`. Start from an overview containing
particles #4, #5 and #6. The entry script processes **#4 -> #5 -> #6**, using
FindParticles IDs sorted top-to-bottom then left-to-right. Before each subsequent
particle, it restores the initial overview position, FoV and defocus. Each particle
has a separate run directory; cancellation stops the entire sequence.
`run_particle_edge_workflow` remains available for individual particles, and
`run_particle_edge_sequence(..., particle_ids=(4, 5, 6))` configures the sequence.

The script fits that particle, traces its edge, and visits 50 nm FoV boxes in
contour order. Edge-centered boxes cover maximal consecutive contour intervals;
multiple loop starts and segment-safe redundant-box removal reduce overlap while
keeping the entire contour covered. This is a heuristic, not a global minimum.
Stage targets use one fixed particle-view reference, so offsets
do not accumulate and the workflow does not return to the center between boxes.
Focus frames are 512 x 512; final photographs are 1024 x 1024. Focus search starts
at +/-4 FoV, then uses +/-0.5 FoV after a converged box. Weak coarse score spans
(<=2%) or other nonconvergence retry once using 3 frames per position and at
least +/-4 FoV. Practical minimum precision is 5 nm. Convergence is a search
criterion and does not guarantee every real image is sharply focused.

Each photograph is analyzed in memory using the frozen normal reference at
`src/scripts/roi_reference.npz`. Positive detections print only:

```text
***ROI Found*** particle #4, box 21: shape variation, texture variation
```

Unconverged final focus and failures print `***Warning***`. The output window
supports plain text; `workflow_messages.html` provides a green ROI/orange
Warning record. The results directory is printed when the run ends.

Results are under
`src/scripts/particle_edge_workflow_results/<timestamp>/particle_NNNN/`:

- Each `edge_box_NNNN/`: raw `haadf.npz`, display `haadf.png`, focus trace,
  zoom/capture metadata, and `roi/` annotations/positions/scores.
- `roi/haadf_roi.png`: candidate bounds, centers, IDs and maximum scores.
- `roi/find_roi.npz`: original image, labels, candidates, descriptions,
  corrected/reference score maps, contextual baseline and threshold map.
- `roi/roi_positions.json`: image-axis offsets and capture stage/FoV/focus.
- `mosaic/`: plain/annotated HAADF mosaic, raw composite/coverage NPZ and ROI
  geometry JSON. Composition uses recorded positions and mean overlap intensity;
  uncovered areas are black. Maximum side is 4096 pixels. It decodes images one
  at a time and does not perform registration/hysteresis correction or ROI
  deduplication across overlapping frames.
- The run root holds `run.json` and `workflow_messages.html`.

On normal completion, the last saved photograph's stage/FoV/focus and profile
parameters are retained for manual Scan, including 1024 x 1024 pixels, dwell
and rotation. `preserve_last_scan=False` restores the initial state instead.
Cancel or an aborted run restores initial stage/FoV/focus. Click Cancel, or
create an empty `STOP` file in the active run folder; cancellation skips mosaic
creation. Driver shutdown may have device-dependent delays.

Capture and ROI-analysis statuses are separate. ROI failure retains a valid
photograph and that frame can still enter the mosaic without ROI annotations.
The run records final profile/state, focus retries and error stage.

## ROI reference and scoring

This is an offline spatial/FFT normal-reference baseline, **not original
PatchCore**. It proposes anomalies and does not identify defect/crystal phases.
No circle-specific shape rule is used. Scores are feature distances, not
probabilities; descriptions summarize contributing feature groups.

Normal patches use a default 5 nm window, 2 nm stride and 0.2 nm analysis pixel
scale. A patch has 34 spatial/intensity/gradient/FFT features. Explicit normal
blur variants estimate feature reliability. A bounded memory preserves diverse
anchors and representative descriptors. Independent normal validation, including
intermediate blur scales, calibrates background/bulk/edge thresholds. Local
same-context median/MAD variation is removed from the reference distance; very
strong reference evidence can bypass that correction. Broad weak anomalies can
still be suppressed, and low-score corner/image-quality false positives remain
possible. Resampling does not create atomic-resolution information.

```python
from algorithms.FindROI import ROIReferenceBank, find_roi_in_image

# Samples are (HAADF array, FoV in meters), explicitly confirmed normal.
reference = ROIReferenceBank().fit(approved_normal_samples)
reference.calibrate(independent_normal_validation_samples)
reference.save("src/scripts/roi_reference.npz")
result = find_roi_in_image(image, fov_m=50e-9, reference=reference)
result.save("roi_output")
```

Include normal corners, orientations, bulk and allowed imaging variations.
Updating the bank invalidates calibration; retain original normal images and
recalibrate. Target images never automatically enter the normal database.
Reference/result format remains v2 (`local_spatial_fft_context_v2`). Local
reference files and acquisition results are ignored by git.

## Validation and review tools

Tests cover detection, physical coordinates, focus confidence/retry,
cancellation/restoration, final scan profile, partial failures, serialization,
layer boundaries and in-memory/lazy processing. The optional local saved-image
regression checks all maps/candidates/descriptions against pre-refactor
fingerprints for 31 HAADF frames; it skips when local data is absent.

`tests/preview_find_roi.py` rebuilds a provisional particle #1 reference and
reviews stored particle #2 photographs under `tests/edge_review_results/`.
Those images and manual-label provenance are local artifacts. Preview results
are not substitutes for validation on real specimens.
# Automatic HAADF brightness and contrast

Import `src/scripts/usim_thin_workflow.py` in Nion Swift to run the complete
window -> first grid cell -> all detected particles -> ranked EELS thickness maps.
After window calibration, Stage Z returns to zero without lens compensation.
Each particle uses coarse autofocus with 512x512 HAADF, one frame per position
and an absolute 100 nm precision target, then maps directly at that measured focus
without additional defocus. `AutofocusConfig` also retains fine mode for imaging.
A focused wider-view HAADF supplies the particle-area footprint; segmented cores
enclosed by the selected region are included, while background holes stay excluded.
Defaults are 128x128 pixels, 10 us/pixel and 2 eV/channel below
500 eV. English progress messages include elapsed time. Results are saved in a
unique `src/scripts/thin_workflow_results/` directory: window labels in `window/`,
thickness maps, spectra and thin-region labels in `particles/particle_XXXX/thickness/`,
and the combined report in `thin_workflow.json`. Descending thin-area rankings
are checkpointed after every particle in `thin_ranking.csv`, `.json` and `.npz`.
Failed or unresolved centers are recorded and excluded from ranking; nonconverged
focus results remain explicitly marked. Cancellation preserves completed results.
Routine truncation/unquantified-area warnings are retained in data, without console
warnings. English progress output includes the current particle and autofocus round.
`thin_mask.png` shows the exact classified area: green thin, gray thick and red
unquantified, without boundary labels obscuring thin pixels. The criterion is
`t/lambda < 1`, not an absolute 100 nm thickness measurement.

`algorithms/AutoBC.py` chooses display limits without clipping any finite pixel
inside the supplied particle footprint. FindThin and FindParticles exports use
these limits; original quantitative arrays and segmentation inputs stay unchanged.
The FindThin and window-to-particle scripts also persist the resulting brightness
and contrast in the selected uSim scan profile. `haadf_autobc.png` is the unlabelled
FindThin preview, and `auto_bc` in the result JSON records the limits. This display
adjustment cannot recover a particle outside the FoV or signal below noise.
