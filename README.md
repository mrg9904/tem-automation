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
