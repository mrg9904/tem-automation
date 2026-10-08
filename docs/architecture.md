# Architecture and contracts

The runtime retains three layers; support modules are inside their owning layer.

| Layer | Owns | Does not own |
| --- | --- | --- |
| adapters | Backend API calls, acquisition contexts, coordinate convention, profile snapshots, device abort | Detection or experiment sequencing |
| algorithms | Image features, reference scoring, contours, fitting, focus search, mosaic composition | Nion UI or experiment status |
| scripts | Experiment policy, target selection, capture/save sequencing, reporting, cancellation/finalization policy | Vendor control details or anomaly-score formulas |

## Device boundary

`Microscope` and `FoVPositioningMicroscope` remain small algorithm protocols.
`WorkflowMicroscope` explicitly declares the additional capabilities needed by
complete workflows. Missing capabilities fail before movement or result-folder
creation. Scan profiles are opaque snapshots: scripts preserve them and only
adapters interpret vendor-specific parameters.

`ImageStageTransform` is the sole implementation of image/stage rotation and
motion sign. The adapter uses it to move; scripts use the same adapter-provided
transform to place saved captures in image coordinates. Distances use meters;
image x points right, image y points down. FoV is the longest image dimension.

Generic cancellation lives in `adapters.cancellation`; the Nion dialog binding
lives in `adapters.nion_usim`. Emergency restore temporarily disables both UI
callbacks and STOP-file checks, then reinstates the previous cancellation state.

## Shared workflow operations

`scripts.common` supplies `CaptureConfig`, `CaptureTarget`, `CaptureResult`,
`FocusPolicy`, `InstrumentState`, `capture_target`, `restore_state`, atomic JSON
writing, completion status and `WorkflowSession` lifecycle management.
`CaptureTarget` is a generic center/size target with particle identity; an edge
box is no longer disguised as a particle dictionary.

The capture result carries the acquired image, device state, focus outcome and
profile snapshot. ROI analysis consumes that image directly, rather than
reopening its NPZ. A successful focus seeds the next box even if ROI analysis
subsequently fails. Capture status and ROI status remain separate; the aggregate
legacy status is retained for consumers. A damaged partial ROI archive is not
loaded into the mosaic. Cancellation during analysis retains the photo while
marking analysis cancelled and restoring the initial device state.

`WorkflowSession` centralizes run directory/checkpoint handling, cancellation,
final state/profile policy and message records. It attempts device restoration
even if an active item's report cannot be saved. Reporter callbacks are passed
from the Nion entry script, so output goes to the script dialog rather than
being lost in a helper module's console.

## ROI modules

`algorithms.FindROI` retains the existing public imports and data format.
Internal `roi` modules separate configuration, feature extraction, reference
memory, detection, result objects, composition and serialization/visualization.
Detection and descriptions share one feature batch and one reference query.
The reference index is reused until reference arrays/weights change.

`compose_roi_mosaic` accepts arrays or lazy image providers with known geometry
and returns `ROIMosaicResult`; computation has no file-path or Nion dependency.
Output helpers read lightweight NPZ geometry, decode each image once when
needed, and export the existing PNG/NPZ/JSON files. Mosaic memory is dominated
by the capped composite/coverage and the current frame, not all source images.

## Behavior-preserving validation

Synthetic and mock-device tests cover ordinary and exceptional paths. Local
pre-refactor fingerprints compare candidates, labels, all score maps and text
for 31 stored images. Reference identity is fixed for this regression; it is
not a test of detection quality for new specimens. No reference retraining or
scoring thresholds are changed by this refactor.
