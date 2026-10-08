"""Nion Swift: sequentially capture edges and detect ROI for particles #4, #5 and #6."""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import numpy as np
from scripts import __file__ as scripts_package_file
from adapters.nion_usim import NionUSimAdapter, nion_cancel_callback
from adapters.cancellation import WorkflowCancelled, check_cancelled, cancellation_scope
from algorithms.FindParticles import FindParticlesConfig, find_particles, find_particles_in_image
from algorithms.FindEdge import FindEdgeConfig, find_edge_in_image
from algorithms.FindROI import ROIReferenceBank, find_roi_in_image, save_roi_mosaic
from algorithms.Zoom2Fit import zoom_to_fit
from algorithms.autofocus import AutofocusConfig
from scripts.common import (CaptureConfig, CaptureTarget, FocusPolicy, InstrumentState,
    WorkflowSession, capture_target, write_json, completion_status, restore_state)


@dataclass(frozen=True)
class ParticleEdgeWorkflowConfig(CaptureConfig):
    padding: float = 1.0
    particle_id: int = 3
    box_fov_m: float = 50e-9
    particle_padding: float = 1.1
    continue_on_error: bool = True
    preserve_last_scan: bool = True
    finding: FindParticlesConfig = field(default_factory=FindParticlesConfig)
    edges: FindEdgeConfig = field(default_factory=FindEdgeConfig)
    autofocus: AutofocusConfig = field(default_factory=lambda: AutofocusConfig(
        initial_half_range_fov_fraction=4.0,points_per_round=5,frames_per_position=1,
        minimum_precision_m=5e-9,minimum_score_span_fraction=.02))
    focus_policy: FocusPolicy = field(default_factory=FocusPolicy)

    def validate(self):
        super().validate()
        self.focus_policy.validate()
        if not isinstance(self.particle_id,int) or isinstance(self.particle_id,bool) or self.particle_id<1:
            raise ValueError('particle_id must be a positive integer')
        if not np.isfinite(self.box_fov_m) or self.box_fov_m<=0:
            raise ValueError('box_fov_m must be finite and positive')
        if not np.isfinite(self.particle_padding) or self.particle_padding<1:
            raise ValueError('particle_padding must be finite and at least 1')


def _prepare_particle_edge(microscope, run_directory, config):
    """Find configured overview particle, fit it, then save its close-up edge and box centers."""
    overview_directory = run_directory / "overview"
    overview_directory.mkdir()
    overview = find_particles(microscope,
        output_path=overview_directory / "find_particles.npz", config=config.finding)
    overview.save_images(overview_directory)
    if not len(overview.particles):
        raise RuntimeError("No particle found in the initial FoV")
    selected = overview.particles[overview.particles['id'] == config.particle_id]
    if not len(selected):
        raise RuntimeError(f'Particle #{config.particle_id} not found in the initial FoV')
    particle = selected[0]
    zoom = zoom_to_fit(microscope,
        (float(particle['offset_x_m']), float(particle['offset_y_m'])),
        float(particle['circle_diameter_m']), padding=config.particle_padding)
    directory = run_directory / f"particle_{config.particle_id:04d}"
    directory.mkdir()
    write_json(directory / "zoom.json", asdict(zoom))
    baseline = InstrumentState.read(microscope)
    image = microscope.acquire_haadf()
    detected = find_particles_in_image(image, fov_m=baseline.fov_m, config=config.edges.particles)
    if not len(detected.particles):
        raise RuntimeError(f"Particle #{config.particle_id} could not be detected in the fitted view")
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
    write_json(directory / "reference.json", {
        "overview_particle_id": config.particle_id, "closeup_particle_id": local_id,
        "overview_particle": {name: particle[name] for name in particle.dtype.names},
        "state": asdict(baseline), "box_count": len(edge.boxes),
        "contour_touches_frame": edge.contour_touches_frame.tolist(),
    })
    return edge, baseline, directory


def _analyze_capture(result,reference,directory,particle_id,box_id):
    roi=find_roi_in_image(result.image,fov_m=result.state.fov_m,reference=reference,
                          output_directory=directory/'roi')
    write_json(directory/'roi/roi_positions.json',{
        'particle_id':particle_id,'box_id':box_id,'capture_state':asdict(result.state),
        'model_id':roi.model_id,'threshold':roi.threshold,
        'coordinate_system':'Offsets from captured FoV center in image axes, x right/y down; not absolute stage coordinates',
        'candidates':[{**{name:item[name] for name in item.dtype.names},'description':roi.descriptions[n]}
                      for n,item in enumerate(roi.candidates)]})
    return roi


def _build_mosaic(run,baseline,transform,particle_directory):
    tiles=[]
    for item in run.manifest['boxes']:
        if 'capture_state' not in item:
            continue
        directory=run.directory/item['directory']
        state=item['capture_state']
        center=transform.image_offset_for_stage_position(state['stage_position_m'],baseline.stage_position_m)
        tiles.append({'capture_path':directory/'haadf.npz','center_m':center,'box_id':item['box_id'],
                      'roi_path':directory/'roi/find_roi.npz' if item.get('roi_status')=='completed' else None})
    if not tiles:
        return
    run.reporter('Building edge HAADF mosaic with ROI labels')
    try:
        metadata=save_roi_mosaic(tiles,particle_directory/'mosaic')
        run.manifest['mosaic']={'status':'completed','directory':str((particle_directory/'mosaic').relative_to(run.directory)),
            'tile_count':metadata['tile_count'],'roi_count':len(metadata['candidates'])}
    except Exception as error:
        run.manifest['mosaic']={'status':'error','error':f'{type(error).__name__}: {error}'}
        raise


def run_particle_edge_workflow(microscope,output_root,*,config=ParticleEdgeWorkflowConfig(),roi_reference=None):
    """Capture and analyze ordered edge boxes; retain last scan on normal completion."""
    config.validate()
    if roi_reference is not None and (roi_reference.threshold is None or
        not np.isfinite(roi_reference.threshold) or roi_reference.threshold<=0):
        raise ValueError('ROI reference must be calibrated')
    with WorkflowSession(microscope,output_root,config,item_key='boxes',count_key='box_count',
        reporter=print,particle_id=config.particle_id,preserve_last_scan=config.preserve_last_scan,
        colored_messages=True) as run:
        run.manifest['roi_reference']=None if roi_reference is None else {
            'model_id':roi_reference.model_id,'threshold':roi_reference.threshold}
        run.checkpoint()
        print(f'Particle #{config.particle_id} edge workflow started')
        edge,baseline,particle_directory=_prepare_particle_edge(microscope,run.directory,config)
        transform=microscope.get_coordinate_transform()
        run.manifest.update(particle_view_state=asdict(baseline),box_count=len(edge.boxes),
            scan_rotation_rad=transform.rotation_rad)
        run.checkpoint()
        print(f'Found {len(edge.boxes)} edge boxes, FoV {config.box_fov_m*1e9:.3f} nm')
        seed_defocus=baseline.defocus_m
        warm_start=False
        for box in edge.boxes:
            check_cancelled()
            microscope.set_defocus(seed_defocus)
            identifier=int(box['id'])
            directory=particle_directory/f'edge_box_{identifier:04d}'
            directory.mkdir()
            item={'box_id':identifier,'particle_id':config.particle_id,
                  'directory':str(directory.relative_to(run.directory)),
                  'box':{name:box[name] for name in box.dtype.names},'roi_status':'pending' if roi_reference is not None else 'not_requested'}
            run.manifest['boxes'].append(item)
            print(f'Autofocus and capture edge box {identifier}/{len(edge.boxes)}')
            phase='capture'
            try:
                focus=config.focus_policy.for_box(config.autofocus,warm_start)
                capture_config=replace(config,autofocus=focus)
                item['focus_start_m']=seed_defocus
                target=CaptureTarget(config.particle_id,(float(box['offset_x_m']),float(box['offset_y_m'])),float(box['fov_m']))
                result=capture_target(microscope,target,directory,capture_config,
                    reference_stage_position_m=baseline.stage_position_m,
                    focus_retry_config=config.focus_policy.retry(focus))
                item.update(result.report)
                if item['autofocus_converged']:
                    seed_defocus=result.state.defocus_m
                    warm_start=True
                else:
                    run.alert(f"***Warning*** particle #{config.particle_id}, box {identifier}: autofocus did not converge ({item['autofocus_stop_reason']}); saved best measured focus",'warning')
                if roi_reference is not None:
                    phase='roi'
                    item['roi_status']='running'
                    check_cancelled()
                    roi=_analyze_capture(result,roi_reference,directory,config.particle_id,identifier)
                    item.update(roi_count=len(roi.candidates),roi_status='completed',
                        roi_directory=str((directory/'roi').relative_to(run.directory)))
                    if len(roi.candidates):
                        descriptions='; '.join(dict.fromkeys(roi.descriptions))
                        run.alert(f'***ROI Found*** particle #{config.particle_id}, box {identifier}: {descriptions}','roi')
            except Exception as error:
                run.record_error(item,error,phase)
                run.alert(f'***Warning*** particle #{config.particle_id}, box {identifier}: {error}','warning')
                write_json(directory/'box.json',item)
                run.checkpoint()
                if not config.continue_on_error:
                    raise
            else:
                write_json(directory/'box.json',item)
                run.checkpoint()
        _build_mosaic(run,baseline,transform,particle_directory)
        run.manifest['status']=completion_status(run.manifest['boxes'])
    print(f"Workflow {run.manifest['status']}")
    return run.directory



def run_particle_edge_sequence(microscope, output_root, *, particle_ids=(4, 5, 6),
                               config=ParticleEdgeWorkflowConfig(), roi_reference=None):
    """Run each particle from the same overview; cancellation stops the sequence."""
    particle_ids = tuple(particle_ids)
    if not particle_ids:
        raise ValueError('particle_ids must not be empty')
    configs = [replace(config, particle_id=identifier) for identifier in particle_ids]
    for particle_config in configs:
        particle_config.validate()
    overview_state = InstrumentState.read(microscope)
    directories = []
    for index, particle_config in enumerate(configs):
        check_cancelled()
        if index:
            restore_state(microscope, overview_state)
            check_cancelled()
        directories.append(run_particle_edge_workflow(
            microscope, output_root, config=particle_config, roi_reference=roi_reference))
    print('Particle edge sequence completed: ' + ', '.join(f'#{identifier}' for identifier in particle_ids))
    return directories


def script_main(api_broker):
    reference_path = Path(scripts_package_file).resolve().parent / 'roi_reference.npz'
    if not reference_path.is_file():
        raise RuntimeError(f'Particle #1 ROI database is missing: {reference_path}')
    reference = ROIReferenceBank.load(reference_path)
    if reference.threshold is None:
        raise RuntimeError('Particle #1 ROI database must be calibrated')
    print('Using fixed particle #1 ROI database')
    api = api_broker.get_api(version="~1.0")
    output_root = Path(scripts_package_file).resolve().parent / "particle_edge_workflow_results"
    with NionUSimAdapter(api, image_size_px=512) as microscope:
        try:
            with cancellation_scope(nion_cancel_callback(print)):
                run_particle_edge_sequence(microscope, output_root, roi_reference=reference)
        except WorkflowCancelled:
            print("Edge workflow stopped; acquired results saved and restoration attempted.")
