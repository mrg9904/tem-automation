"""Nion Swift: identify particles and orchestrate fit/focus/capture operations."""
from dataclasses import dataclass, field
from adapters.nion_usim import NionUSimAdapter, nion_cancel_callback
from adapters.cancellation import WorkflowCancelled, check_cancelled, cancellation_scope
from algorithms.FindParticles import FindParticlesConfig, find_particles
from scripts import __file__ as scripts_package_file
from pathlib import Path
from scripts.common import (CaptureConfig, CaptureTarget, WorkflowSession, capture_target,
                            restore_state, write_json, completion_status)


@dataclass(frozen=True)
class ParticleWorkflowConfig(CaptureConfig):
    continue_on_error: bool = True
    finding: FindParticlesConfig = field(default_factory=FindParticlesConfig)


def run_particle_workflow(microscope, output_root, *, config=ParticleWorkflowConfig()):
    config.validate()
    with WorkflowSession(microscope,output_root,config,item_key='particles',count_key='particle_count',reporter=print) as run:
        print(f'Particle workflow results: {run.directory}')
        overview_directory=run.directory/'overview'
        overview_directory.mkdir()
        overview=find_particles(microscope,output_path=overview_directory/'find_particles.npz',config=config.finding)
        overview.save_images(overview_directory)
        run.manifest['particle_count']=len(overview.particles)
        run.checkpoint()
        print(f'Detected {len(overview.particles)} particles')
        for particle in overview.particles:
            check_cancelled()
            restore_state(microscope,run.initial)
            identifier=int(particle['id'])
            directory=run.directory/f'particle_{identifier:04d}'
            directory.mkdir()
            entry={'id':identifier,'directory':directory.name,
                   'overview_particle':{name:particle[name] for name in particle.dtype.names}}
            run.manifest['particles'].append(entry)
            print(f'Processing particle {identifier}/{len(overview.particles)}')
            target=CaptureTarget(identifier,(float(particle['offset_x_m']),float(particle['offset_y_m'])),float(particle['circle_diameter_m']))
            try:
                result=capture_target(microscope,target,directory,config)
                entry.update(result.report)
            except Exception as error:
                run.record_error(entry,error,'capture')
                write_json(directory/'particle.json',entry)
                run.checkpoint()
                print(f'Particle {identifier} failed: {error}')
                if not config.continue_on_error:
                    raise
            else:
                write_json(directory/'particle.json',entry)
                run.checkpoint()
                if entry['status']=='captured_unconverged':
                    print(f'Particle {identifier}: captured best measured focus; autofocus did not converge')
        run.manifest['status']=completion_status(run.manifest['particles'])
    print(f"Workflow {run.manifest['status']}: {run.directory}")
    return run.directory


def script_main(api_broker):
    api = api_broker.get_api(version="~1.0")
    output_root = Path(scripts_package_file).resolve().parent / "particle_workflow_results"
    with NionUSimAdapter(api, image_size_px=512) as microscope:
        try:
            with cancellation_scope(nion_cancel_callback(print)):
                run_particle_workflow(microscope, output_root)
        except WorkflowCancelled:
            print("Workflow stopped; acquired results saved and restoration attempted.")
