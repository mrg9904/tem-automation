"""Nion Swift: fit overview particle #1, find its edge, and capture 50 nm boxes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from scripts import __file__ as scripts_package_file
from adapters.nion_usim import NionUSimAdapter
from adapters.cancellation import (WorkflowCancelled, check_cancelled, set_stop_file, cancellation_scope, nion_cancel_callback)
from algorithms.FindParticles import FindParticlesConfig, find_particles, find_particles_in_image
from algorithms.FindEdge import FindEdgeConfig, find_edge_in_image
from algorithms.Zoom2Fit import zoom_to_fit
from algorithms.autofocus import AutofocusConfig
from scripts.usim_particle_workflow import (
    InstrumentState, ParticleWorkflowConfig, _process_particle, _restore_state, _write_json,
)


@dataclass(frozen=True)
class ParticleEdgeWorkflowConfig:
    box_fov_m: float = 50e-9
    particle_padding: float = 1.1
    continue_on_error: bool = True
    finding: FindParticlesConfig = field(default_factory=FindParticlesConfig)
    edges: FindEdgeConfig = field(default_factory=FindEdgeConfig)
    autofocus: AutofocusConfig = field(default_factory=lambda: AutofocusConfig(initial_half_range_fov_fraction=4.0, points_per_round=5,
        frames_per_position=1, minimum_precision_m=5e-9))


def _prepare_particle_edge(microscope, run_directory, config):
    """Find overview #1, fit it, then save its close-up edge and box centers."""
    overview_directory = run_directory / "overview"
    overview_directory.mkdir()
    overview = find_particles(microscope,
        output_path=overview_directory / "find_particles.npz", config=config.finding)
    overview.save_images(overview_directory)
    if not len(overview.particles):
        raise RuntimeError("No particle found in the initial FoV")
    particle = overview.particles[overview.particles['id'] == 1][0]
    zoom = zoom_to_fit(microscope,
        (float(particle['offset_x_m']), float(particle['offset_y_m'])),
        float(particle['circle_diameter_m']), padding=config.particle_padding)
    directory = run_directory / "particle_0001"
    directory.mkdir()
    _write_json(directory / "zoom.json", asdict(zoom))
    baseline = InstrumentState.read(microscope)
    image = microscope.acquire_haadf()
    detected = find_particles_in_image(image, fov_m=baseline.fov_m, config=config.edges.particles)
    if not len(detected.particles):
        raise RuntimeError("Particle #1 could not be detected in the fitted view")
    # Close-up IDs differ from overview IDs; select the region at/nearest the image center.
    height, width = detected.image.shape
    local_id = int(detected.labels[height // 2, width // 2])
    if not local_id:
        distances = detected.particles['offset_x_m']**2 + detected.particles['offset_y_m']**2
        local_id = int(detected.particles[np.argmin(distances)]['id'])
    edge = find_edge_in_image(image, fov_m=baseline.fov_m, box_fov_m=config.box_fov_m,
                              particle_id=local_id, config=config.edges)
    edge.save(directory / "find_edge.npz")
    edge.save_overlay(directory / "haadf_edge_boxes.png")
    _write_json(directory / "reference.json", {
        "overview_particle_id": 1, "closeup_particle_id": local_id,
        "overview_particle": {name: particle[name] for name in particle.dtype.names},
        "state": asdict(baseline), "box_count": len(edge.boxes),
        "contour_touches_frame": edge.contour_touches_frame.tolist(),
    })
    return edge, baseline, directory


def _capture_edge_box(microscope, box, directory, config, *, reference_stage_position_m=None):
    # Compose the shared Zoom2Fit -> autofocus -> final capture pipeline.
    # id remains the original particle ID; box IDs are recorded separately.
    target = {"id": 1, "offset_x_m": box['offset_x_m'], "offset_y_m": box['offset_y_m'],
              "circle_diameter_m": box['fov_m']}
    capture_config = ParticleWorkflowConfig(padding=1.0,
        continue_on_error=config.continue_on_error, autofocus=config.autofocus)
    return _process_particle(microscope, target, directory, capture_config,
        reference_stage_position_m=reference_stage_position_m)


def run_particle_edge_workflow(microscope, output_root, *, config=ParticleEdgeWorkflowConfig()):
    """Capture every 50 nm edge box from a single particle-view baseline.

    The starting view must be an overview containing the intended particle #1
    (FindParticles numbering: top-to-bottom, then left-to-right). No saved
    previous run is needed. The initial stage/FoV/defocus are restored on exit.
    """
    if not np.isfinite(config.box_fov_m) or config.box_fov_m <= 0:
        raise ValueError("box_fov_m must be finite and positive")
    if not np.isfinite(config.particle_padding) or config.particle_padding < 1:
        raise ValueError("particle_padding must be finite and at least 1")
    initial = InstrumentState.read(microscope)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    run_directory = Path(output_root) / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "particle_id": 1,
                "initial_state": asdict(initial), "config": asdict(config), "box_count": 0, "boxes": []}
    path = run_directory / "run.json"
    _write_json(path, manifest)
    print(f"Particle #1 edge workflow results: {run_directory}")
    previous_stop_file = set_stop_file(run_directory / "STOP")
    try:
        edge, baseline, particle_directory = _prepare_particle_edge(microscope, run_directory, config)
        manifest.update(particle_view_state=asdict(baseline), box_count=len(edge.boxes))
        _write_json(path, manifest)
        print(f"Found {len(edge.boxes)} edge boxes, FoV {config.box_fov_m * 1e9:.3f} nm")
        seed_defocus = baseline.defocus_m
        warm_start = False
        for box in edge.boxes:
            check_cancelled()
            # Offsets are anchored to the reference but stage moves directly
            # around the contour; never visit the reference between boxes.
            microscope.set_defocus(seed_defocus)
            box_id = int(box['id'])
            directory = particle_directory / f"edge_box_{box_id:04d}"
            directory.mkdir()
            entry = {"box_id": box_id, "particle_id": 1,
                     "directory": str(directory.relative_to(run_directory)),
                     "box": {name: box[name] for name in box.dtype.names}}
            manifest['boxes'].append(entry)
            print(f"Autofocus and capture edge box {box_id}/{len(edge.boxes)}")
            try:
                focus_config = (replace(config.autofocus, initial_half_range_fov_fraction=.5)
                                if warm_start else config.autofocus)
                box_config = replace(config, autofocus=focus_config)
                entry['focus_start_m'] = seed_defocus
                entry.update(_capture_edge_box(microscope, box, directory, box_config,
                    reference_stage_position_m=baseline.stage_position_m))
                if entry['autofocus_converged']:
                    seed_defocus = entry['capture_state']['defocus_m']
                    warm_start = True
            except Exception as error:
                entry.update(status="error", error=f"{type(error).__name__}: {error}")
                _write_json(directory / "box.json", entry)
                _write_json(path, manifest)
                print(f"Box {box_id} failed: {error}")
                if not config.continue_on_error:
                    raise
            else:
                _write_json(directory / "box.json", entry)
                _write_json(path, manifest)
                if entry['status'] == 'captured_unconverged':
                    print(f"Box {box_id}: captured best measured focus; autofocus did not converge")
        statuses = [entry['status'] for entry in manifest['boxes']]
        manifest['status'] = ('completed_with_errors' if 'error' in statuses else
                              'completed_with_unconverged_focus' if 'captured_unconverged' in statuses else 'completed')
    except BaseException as error:
        for active in manifest['boxes']:
            if 'status' not in active:
                active['status'] = 'cancelled' if isinstance(error, WorkflowCancelled) else 'aborted'
                _write_json(run_directory / active['directory'] / 'box.json', active)
        manifest.update(status='cancelled' if isinstance(error, WorkflowCancelled) else 'aborted', error=f"{type(error).__name__}: {error}")
        raise
    finally:
        set_stop_file(previous_stop_file)
        try:
            _restore_state(microscope, initial)
            manifest['state_restored'] = True
        except Exception as error:
            manifest.update(status='restoration_failed', state_restored=False, restoration_error=str(error))
            raise
        finally:
            _write_json(path, manifest)
    print(f"Workflow {manifest['status']}: {run_directory}")
    return run_directory


def script_main(api_broker):
    api = api_broker.get_api(version="~1.0")
    output_root = Path(scripts_package_file).resolve().parent / "particle_edge_workflow_results"
    with NionUSimAdapter(api, image_size_px=512) as microscope:
        try:
            with cancellation_scope(nion_cancel_callback(print)):
                run_particle_edge_workflow(microscope, output_root)
        except WorkflowCancelled:
            print("Edge workflow stopped; acquired results saved and restoration attempted.")
