"""Shared workflow operations, persistence and lifecycle; no Nion API access."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import html
import json
import numpy as np
from PIL import Image
from adapters.base import require_workflow_microscope
from adapters.cancellation import WorkflowCancelled, check_cancelled, set_stop_file, suspend_cancellation
from algorithms.Zoom2Fit import zoom_to_fit
from algorithms.autofocus import AutofocusConfig, autofocus


@dataclass(frozen=True)
class CaptureConfig:
    padding: float = 1.1
    autofocus_image_size_px: int = 512
    capture_image_size_px: int = 1024
    dwell_time_us: float = 1.0
    autofocus: AutofocusConfig = field(default_factory=AutofocusConfig)

    def validate(self):
        self.autofocus.validate()
        if not np.isfinite(self.padding) or self.padding<1:
            raise ValueError('padding must be finite and at least 1')
        for size in (self.autofocus_image_size_px,self.capture_image_size_px):
            if not isinstance(size,int) or isinstance(size,bool) or size<=1:
                raise ValueError('Image sizes must be integers greater than one')
        if not np.isfinite(self.dwell_time_us) or self.dwell_time_us<=0:
            raise ValueError('dwell_time_us must be finite and positive')


@dataclass(frozen=True)
class FocusPolicy:
    warm_start_half_range_fov_fraction: float = .5
    retry_half_range_fov_fraction: float = 4.0
    retry_frames_per_position: int = 3

    def validate(self):
        if any(not np.isfinite(v) or v<=0 for v in
            (self.warm_start_half_range_fov_fraction,self.retry_half_range_fov_fraction)):
            raise ValueError('Focus search ranges must be finite and positive')
        if not isinstance(self.retry_frames_per_position,int) or isinstance(self.retry_frames_per_position,bool) or self.retry_frames_per_position<1:
            raise ValueError('Retry frame count must be a positive integer')

    def for_box(self,config,warm_start):
        return replace(config,initial_half_range_fov_fraction=self.warm_start_half_range_fov_fraction) if warm_start else config

    def retry(self,config):
        return replace(config,frames_per_position=self.retry_frames_per_position,
            initial_half_range_fov_fraction=max(self.retry_half_range_fov_fraction,config.initial_half_range_fov_fraction))


@dataclass(frozen=True)
class CaptureTarget:
    particle_id: int
    center_offset_m: tuple[float,float]
    diameter_m: float


@dataclass(frozen=True)
class InstrumentState:
    stage_position_m: tuple[float,float]
    fov_m: float
    defocus_m: float

    @classmethod
    def read(cls,microscope):
        state=cls(microscope.get_stage_position(),microscope.get_fov(),microscope.get_defocus())
        if not np.isfinite((*state.stage_position_m,state.fov_m,state.defocus_m)).all() or state.fov_m<=0:
            raise RuntimeError('Invalid initial microscope state')
        return state


@dataclass(frozen=True)
class CaptureResult:
    image: np.ndarray
    state: InstrumentState
    report: dict


def restore_state(microscope,state):
    errors=[]
    with suspend_cancellation():
        for name,operation in (
            ('stage',lambda:microscope.set_stage_position(*state.stage_position_m)),
            ('FoV',lambda:microscope.set_fov(state.fov_m)),
            ('defocus',lambda:microscope.set_defocus(state.defocus_m))):
            try:
                operation()
            except Exception as error:
                errors.append(f'{name}: {error}')
    if errors:
        raise RuntimeError('Could not restore overview state: '+'; '.join(errors))


def json_value(value):
    if isinstance(value,dict):
        return {key:json_value(item) for key,item in value.items()}
    if isinstance(value,np.ndarray):
        return json_value(value.tolist())
    if isinstance(value,(list,tuple)):
        return [json_value(item) for item in value]
    if isinstance(value,np.generic):
        return json_value(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def write_json(path,data):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_value(data),indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def save_capture(image,state,directory,particle_id):
    np.savez_compressed(directory/'haadf.npz',image=image,particle_id=np.int32(particle_id),
        fov_m=np.float64(state.fov_m),stage_position_m=np.asarray(state.stage_position_m),
        defocus_m=np.float64(state.defocus_m),pixel_size_m=np.float64(state.fov_m/max(image.shape)))
    low,high=float(image.min()),float(image.max())
    pixels=(np.rint(np.clip((image.astype(np.float64)-low)/(high-low),0,1)*255).astype(np.uint8)
        if high>low else np.zeros(image.shape,dtype=np.uint8))
    Image.fromarray(pixels).save(directory/'haadf.png')


def capture_target(microscope,target,directory,config,*,reference_stage_position_m=None,focus_retry_config=None):
    config.validate()
    check_cancelled()
    zoom=zoom_to_fit(microscope,target.center_offset_m,target.diameter_m,padding=config.padding,
                    reference_stage_position_m=reference_stage_position_m)
    write_json(directory/'zoom.json',asdict(zoom))
    with microscope.acquisition_settings(image_size_px=config.autofocus_image_size_px,dwell_time_us=config.dwell_time_us):
        focus=autofocus(microscope,config=config.autofocus)
        retried=False
        if not focus.converged and focus_retry_config is not None:
            write_json(directory/'autofocus_initial.json',asdict(focus))
            check_cancelled()
            retried=True
            focus=autofocus(microscope,config=focus_retry_config)
    write_json(directory/'autofocus.json',asdict(focus))
    np.savez_compressed(directory/'autofocus.npz',
        defocus_m=np.asarray([item.defocus_m for item in focus.measurements]),
        score=np.asarray([item.score for item in focus.measurements]),
        best_defocus_m=np.float64(focus.best_defocus_m),converged=np.bool_(focus.converged))
    with microscope.acquisition_settings(image_size_px=config.capture_image_size_px,dwell_time_us=config.dwell_time_us):
        image=np.asarray(microscope.acquire_haadf(),dtype=np.float32)
        state=InstrumentState.read(microscope)
        profile=microscope.get_last_scan_profile()
    if image.ndim!=2 or not image.size or not np.isfinite(image).all():
        raise RuntimeError('Invalid final HAADF image')
    save_capture(image,state,directory,target.particle_id)
    status='captured' if focus.converged else 'captured_unconverged'
    return CaptureResult(image,state,{'status':status,'capture_status':status,
        'autofocus_converged':focus.converged,'autofocus_stop_reason':focus.stop_reason,
        'autofocus_retried':retried,'capture_state':asdict(state),'capture_scan_profile':profile})


def completion_status(entries):
    if any(entry.get('status')=='error' for entry in entries):
        return 'completed_with_errors'
    if any(entry.get('capture_status',entry.get('status'))=='captured_unconverged' for entry in entries):
        return 'completed_with_unconverged_focus'
    return 'completed'


class WorkflowSession:
    """Checkpoint, cancellation and final device policy shared by both workflows."""
    def __init__(self,microscope,output_root,config,*,item_key,count_key,reporter=print,
                 particle_id=None,preserve_last_scan=False,colored_messages=False):
        self.microscope=microscope
        self.output_root=Path(output_root)
        self.config=config
        self.item_key=item_key
        self.count_key=count_key
        self.reporter=reporter
        self.particle_id=particle_id
        self.preserve_last_scan=preserve_last_scan
        self.colored_messages=colored_messages

    def __enter__(self):
        require_workflow_microscope(self.microscope)
        self.initial=InstrumentState.read(self.microscope)
        name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8]
        self.directory=self.output_root/name
        self.directory.mkdir(parents=True,exist_ok=False)
        self.manifest={'schema_version':1,'status':'running','initial_state':asdict(self.initial),
            'config':asdict(self.config),self.count_key:0,self.item_key:[]}
        if self.particle_id is not None:
            self.manifest['particle_id']=self.particle_id
        self.checkpoint()
        self.previous_stop_file=set_stop_file(self.directory/'STOP')
        return self

    def checkpoint(self):
        write_json(self.directory/'run.json',self.manifest)

    def alert(self,text,kind):
        self.reporter(text)
        self.manifest.setdefault('messages',[]).append({'kind':kind,'text':text})

    def record_error(self,entry,error,stage):
        entry.update(status='error',error_stage=stage,error=f'{type(error).__name__}: {error}')
        entry['roi_status' if stage=='roi' else 'capture_status']='error'
        if stage=='capture' and entry.get('roi_status')=='pending':
            entry['roi_status']='skipped'

    def _save_messages(self):
        lines=['<!doctype html><meta charset="utf-8"><title>Workflow messages</title><body style="background:#fff;font:16px monospace">']
        for message in self.manifest.get('messages',[]):
            color='#16803c' if message['kind']=='roi' else '#b65b00'
            lines.append(f'<p style="color:{color}">{html.escape(message["text"])}</p>')
        lines.append('</body>')
        (self.directory/'workflow_messages.html').write_text('\n'.join(lines),encoding='utf-8')

    def __exit__(self,exception_type,error,traceback):
        if error is not None:
            status='cancelled' if isinstance(error,WorkflowCancelled) else 'aborted'
            for active in self.manifest[self.item_key]:
                if 'status' not in active or active.get('roi_status')=='running':
                    active['status']=status
                    if active.get('roi_status')=='running':
                        active['roi_status']=status
                    else:
                        active['capture_status']=status
                    filename='box.json' if self.item_key=='boxes' else 'particle.json'
                    try:
                        write_json(self.directory/active['directory']/filename,active)
                    except Exception as report_error:
                        self.manifest.setdefault('report_errors',[]).append(str(report_error))
            self.manifest.update(status=status,error=f'{type(error).__name__}: {error}')
        set_stop_file(self.previous_stop_file)
        try:
            captures=[entry for entry in self.manifest[self.item_key] if 'capture_state' in entry]
            completed=self.manifest['status'].startswith('completed')
            if self.preserve_last_scan and completed and captures:
                state=InstrumentState(**captures[-1]['capture_state'])
                restore_state(self.microscope,state)
                with suspend_cancellation():
                    self.manifest['final_scan_profile']=self.microscope.persist_last_scan_profile(captures[-1]['capture_scan_profile'])
                self.manifest.update(final_state=asdict(state),state_restored=False,last_scan_preserved=True)
            else:
                restore_state(self.microscope,self.initial)
                self.manifest['state_restored']=True
                if self.item_key=='boxes':
                    self.manifest['last_scan_preserved']=False
        except Exception as restore_error:
            self.manifest.update(status='restoration_failed',state_restored=False,restoration_error=str(restore_error))
            raise
        finally:
            self.checkpoint()
            if self.colored_messages:
                self._save_messages()
                self.reporter(f'Images and data saved to: {self.directory}')
        return False
