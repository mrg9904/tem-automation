"""Nion Swift workflow: find particles, center, autofocus, and record each one."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from scripts import __file__ as scripts_package_file
from adapters.nion_usim import NionUSimAdapter
from algorithms.FindParticles import FindParticlesConfig, find_particles
from algorithms.Zoom2Fit import zoom_to_fit
from algorithms.autofocus import AutofocusConfig, autofocus


@dataclass(frozen=True)
class ParticleWorkflowConfig:
    padding: float = 1.1
    continue_on_error: bool = True
    finding: FindParticlesConfig = field(default_factory=FindParticlesConfig)
    autofocus: AutofocusConfig = field(default_factory=AutofocusConfig)


@dataclass(frozen=True)
class InstrumentState:
    stage_position_m: tuple[float, float]
    fov_m: float
    defocus_m: float

    @classmethod
    def read(cls, microscope):
        state = cls(microscope.get_stage_position(), microscope.get_fov(), microscope.get_defocus())
        if not np.all(np.isfinite((*state.stage_position_m, state.fov_m, state.defocus_m))) or state.fov_m <= 0:
            raise RuntimeError("Invalid initial microscope state")
        return state


def _restore_state(microscope, state):
    """Attempt all restorations, even if one setting fails."""
    errors = []
    for name, restore in (
        ("stage", lambda: microscope.set_stage_position(*state.stage_position_m)),
        ("FoV", lambda: microscope.set_fov(state.fov_m)),
        ("defocus", lambda: microscope.set_defocus(state.defocus_m)),
    ):
        try:
            restore()
        except Exception as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise RuntimeError("Could not restore overview state: " + "; ".join(errors))


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, data):
    # Replace only after a complete JSON write, preserving the last checkpoint.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(_json_value(data), indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(path)


def _capture_overview(microscope, directory, config):
    directory.mkdir()
    result = find_particles(microscope, output_path=directory / "find_particles.npz", config=config.finding)
    result.save_images(directory)
    return result


def _save_capture(microscope, directory, particle_id):
    """Save quantitative image data separately from the display-scaled PNG."""
    image = np.asarray(microscope.acquire_haadf(), dtype=np.float32)
    if image.ndim != 2 or not image.size or not np.all(np.isfinite(image)):
        raise RuntimeError("Invalid final HAADF image")
    state = InstrumentState.read(microscope)
    np.savez_compressed(directory / "haadf.npz", image=image,
        particle_id=np.int32(particle_id), fov_m=np.float64(state.fov_m),
        stage_position_m=np.asarray(state.stage_position_m), defocus_m=np.float64(state.defocus_m),
        pixel_size_m=np.float64(state.fov_m / max(image.shape)))
    low, high = float(image.min()), float(image.max())
    pixels = (np.rint(np.clip((image.astype(np.float64) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
              if high > low else np.zeros(image.shape, dtype=np.uint8))
    Image.fromarray(pixels).save(directory / "haadf.png")
    return state


def _process_particle(microscope, particle, directory, config):
    """Compose the positioning and autofocus algorithms for one particle."""
    zoom = zoom_to_fit(microscope,
        (float(particle["offset_x_m"]), float(particle["offset_y_m"])),
        float(particle["circle_diameter_m"]), padding=config.padding)
    _write_json(directory / "zoom.json", asdict(zoom))
    focus = autofocus(microscope, config=config.autofocus)
    _write_json(directory / "autofocus.json", asdict(focus))
    np.savez_compressed(directory / "autofocus.npz",
        defocus_m=np.asarray([measurement.defocus_m for measurement in focus.measurements]),
        score=np.asarray([measurement.score for measurement in focus.measurements]),
        best_defocus_m=np.float64(focus.best_defocus_m), converged=np.bool_(focus.converged))
    state = _save_capture(microscope, directory, int(particle["id"]))
    return {"status": "captured" if focus.converged else "captured_unconverged",
            "autofocus_converged": focus.converged, "capture_state": asdict(state)}


def run_particle_workflow(microscope, output_root, *, config=ParticleWorkflowConfig()):
    """Run from the current view; restore its stage/FoV/focus on exit.

    Every particle is positioned relative to the same overview state, so
    offsets never accumulate. An unconverged focus still captures the best
    measured position and is flagged. Instrument restoration failures abort
    the run rather than proceeding with an unknown coordinate baseline.
    """
    if not np.isfinite(config.padding) or config.padding < 1:
        raise ValueError("padding must be finite and at least 1")
    initial = InstrumentState.read(microscope)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    run_directory = Path(output_root) / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "initial_state": asdict(initial),
                "config": asdict(config), "particle_count": 0, "particles": []}
    manifest_path = run_directory / "run.json"
    _write_json(manifest_path, manifest)
    print(f"Particle workflow results: {run_directory}")
    try:
        overview = _capture_overview(microscope, run_directory / "overview", config)
        manifest["particle_count"] = len(overview.particles)
        _write_json(manifest_path, manifest)
        print(f"Detected {len(overview.particles)} particles")
        for particle in overview.particles:
            # Re-establish the acquisition baseline before applying overview offsets.
            _restore_state(microscope, initial)
            particle_id = int(particle["id"])
            directory = run_directory / f"particle_{particle_id:04d}"
            directory.mkdir()
            entry = {"id": particle_id, "directory": directory.name,
                     "overview_particle": {name: particle[name] for name in particle.dtype.names}}
            manifest["particles"].append(entry)
            print(f"Processing particle {particle_id}/{len(overview.particles)}")
            try:
                entry.update(_process_particle(microscope, particle, directory, config))
            except Exception as error:
                entry.update(status="error", error=f"{type(error).__name__}: {error}")
                _write_json(directory / "particle.json", entry)
                _write_json(manifest_path, manifest)
                print(f"Particle {particle_id} failed: {error}")
                if not config.continue_on_error:
                    raise
            else:
                _write_json(directory / "particle.json", entry)
                _write_json(manifest_path, manifest)
                if entry["status"] == "captured_unconverged":
                    print(f"Particle {particle_id}: captured best measured focus; autofocus did not converge")
        statuses = [item["status"] for item in manifest["particles"]]
        manifest["status"] = ("completed_with_errors" if "error" in statuses else
                              "completed_with_unconverged_focus" if "captured_unconverged" in statuses else "completed")
    except BaseException as error:
        manifest.update(status="aborted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            _restore_state(microscope, initial)
            manifest["state_restored"] = True
        except Exception as error:
            manifest.update(status="restoration_failed", state_restored=False, restoration_error=str(error))
            raise
        finally:
            _write_json(manifest_path, manifest)
    print(f"Workflow {manifest['status']}: {run_directory}")
    return run_directory


def script_main(api_broker):
    api = api_broker.get_api(version="~1.0")
    output_root = Path(scripts_package_file).resolve().parent / "particle_workflow_results"
    with NionUSimAdapter(api, image_size_px=512) as microscope:
        run_particle_workflow(microscope, output_root)
