"""Layer boundaries and shared workflow contracts."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from adapters.coordinates import ImageStageTransform
from adapters.cancellation import WorkflowCancelled, check_cancelled, set_stop_file, suspend_cancellation
from scripts.common import CaptureConfig, WorkflowSession
from tests.test_particle_workflow import WorkflowMicroscope, focus_result
from tests.test_particle_edge_workflow import EdgeMicroscope
from tests import test_particle_edge_workflow as edge_fixtures
from scripts import usim_particle_edge_workflow as edge


class ArchitectureTest(unittest.TestCase):
    def test_imports_do_not_reverse_layer_boundaries(self):
        src=Path(__file__).resolve().parents[1]/'src'
        for layer in ('adapters','algorithms','scripts'):
            for path in (src/layer).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    modules=([node.module or ''] if isinstance(node,ast.ImportFrom)
                             else [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
                    for module in modules:
                        if layer=='adapters':
                            self.assertFalse(module.startswith(('algorithms','scripts')),str(path))
                        elif layer=='algorithms':
                            self.assertFalse(module.startswith(('scripts','nion')),str(path))
                            self.assertNotEqual(module,'adapters.nion_usim',str(path))
                        else:
                            self.assertFalse(module.startswith('scripts.usim_'),str(path))

    def test_coordinates_roundtrip_both_stage_directions_and_rotations(self):
        for angle in (0.,.7,np.pi/2,np.pi):
            for direction in (-1.,1.):
                transform=ImageStageTransform(angle,direction)
                reference=(120e-9,-50e-9)
                offset=(30e-9,12e-9)
                stage=transform.stage_position_for_offset(offset,reference)
                np.testing.assert_allclose(transform.image_offset_for_stage_position(stage,reference),offset,atol=1e-22)

    def test_missing_device_capability_fails_before_movement_or_directory_creation(self):
        microscope=WorkflowMicroscope()
        microscope.get_coordinate_transform=None
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(TypeError,'get_coordinate_transform'):
                with WorkflowSession(microscope,root,CaptureConfig(),item_key='boxes',count_key='box_count'):
                    self.fail('should not enter')
            self.assertEqual(list(Path(root).iterdir()),[])
            self.assertEqual(microscope.moves,[])

    def test_emergency_restore_suspends_stop_file_and_resumes_it_afterwards(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'STOP';path.touch()
            previous=set_stop_file(path)
            try:
                with suspend_cancellation():
                    check_cancelled()
                with self.assertRaises(WorkflowCancelled):
                    check_cancelled()
            finally:
                set_stop_file(previous)

    def test_roi_failure_keeps_capture_status_and_preserves_final_scan(self):
        from dataclasses import replace
        microscope=EdgeMicroscope()
        reference=mock.Mock(threshold=.1,model_id='fixed')
        def partial_roi(image,**kwargs):
            output=Path(kwargs['output_directory']);output.mkdir(parents=True)
            (output/'find_roi.npz').write_bytes(b'incomplete archive')
            raise RuntimeError('analysis failed')
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus',side_effect=focus_result), mock.patch.object(edge,'find_roi_in_image',side_effect=partial_roi), mock.patch('builtins.print'):
            run=edge.run_particle_edge_workflow(microscope,root,config=replace(edge_fixtures.ParticleEdgeWorkflowTest.config,preserve_last_scan=True),roi_reference=reference)
            report=json.loads((run/'run.json').read_text())
            self.assertEqual(report['status'],'completed_with_errors')
            self.assertTrue(report['last_scan_preserved'])
            for entry in report['boxes']:
                self.assertEqual(entry['capture_status'],'captured')
                self.assertEqual(entry['roi_status'],'error')
                self.assertEqual(entry['error_stage'],'roi')
                self.assertTrue((run/entry['directory']/'haadf.npz').is_file())
            self.assertEqual(report['mosaic']['tile_count'],report['box_count'])

    def test_cancel_during_roi_restores_initial_state_even_after_capture(self):
        microscope=EdgeMicroscope()
        reference=mock.Mock(threshold=.1,model_id='fixed')
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus',side_effect=focus_result), mock.patch.object(edge,'find_roi_in_image',side_effect=WorkflowCancelled('stop')), mock.patch('builtins.print'):
            with self.assertRaises(WorkflowCancelled):
                edge.run_particle_edge_workflow(microscope,root,config=edge_fixtures.ParticleEdgeWorkflowTest.config,roi_reference=reference)
            report=json.loads((next(Path(root).iterdir())/'run.json').read_text())
            self.assertEqual(report['status'],'cancelled')
            self.assertTrue(report['state_restored'])
            self.assertEqual(report['boxes'][0]['roi_status'],'cancelled')
            self.assertEqual(report['boxes'][0]['capture_status'],'captured')
            self.assertEqual(microscope.stage,(1222e-9,279e-9))

    def test_roi_analysis_uses_in_memory_capture_without_reading_npz(self):
        from algorithms.FindROI import FindROIConfig, ROIReferenceBank
        from scripts.common import CaptureResult, InstrumentState
        y,x=np.mgrid[:40,:40]
        image=(x>=20).astype(np.float32)
        bank=ROIReferenceBank(FindROIConfig(analysis_pixel_size_nm=1.,patch_size_nm=5.,stride_nm=5.))
        bank.fit([(image,40e-9)])
        bank.calibrate([(image,40e-9)])
        result=CaptureResult(image,InstrumentState((0.,0.),40e-9,0.),{})
        with tempfile.TemporaryDirectory() as directory, mock.patch('numpy.load',side_effect=AssertionError('unexpected image reread')):
            roi=edge._analyze_capture(result,bank,Path(directory),3,1)
            self.assertEqual(len(roi.candidates),0)
            self.assertTrue((Path(directory)/'roi/haadf_roi.png').is_file())

    def test_cancel_still_restores_when_active_box_report_cannot_be_written(self):
        from scripts import common
        microscope=WorkflowMicroscope()
        original=common.write_json
        def fail_box(path,data):
            if Path(path).name=='box.json':
                raise OSError('box report unavailable')
            return original(path,data)
        with tempfile.TemporaryDirectory() as root, mock.patch.object(common,'write_json',side_effect=fail_box):
            with self.assertRaises(WorkflowCancelled):
                with WorkflowSession(microscope,root,CaptureConfig(),item_key='boxes',count_key='box_count') as session:
                    (session.directory/'active').mkdir()
                    session.manifest['boxes'].append({'directory':'active'})
                    microscope.set_stage_position(0.,0.)
                    raise WorkflowCancelled('stop')
            report=json.loads((session.directory/'run.json').read_text())
            self.assertTrue(report['state_restored'])
            self.assertEqual(report['status'],'cancelled')
            self.assertIn('box report unavailable',report['report_errors'])
            self.assertEqual(microscope.stage,(1222e-9,279e-9))

    def test_numpy_scalar_array_metadata_is_json_serializable(self):
        from scripts.common import write_json
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'metadata.json'
            write_json(path,{'scalar':np.array(2.),'vector':np.array([1,2])})
            self.assertEqual(json.loads(path.read_text()),{'scalar':2.,'vector':[1,2]})

    def test_rotated_workflow_mosaic_uses_the_same_geometry_as_stage_movement(self):
        class RotatedMicroscope(EdgeMicroscope):
            def get_coordinate_transform(self):
                return ImageStageTransform(np.pi/2)
            def center_fov_on_image_offset(self,x,y,*,reference_stage_position_m=None):
                reference=self.stage if reference_stage_position_m is None else reference_stage_position_m
                self.stage=tuple(self.get_coordinate_transform().stage_position_for_offset((x,y),reference))
                self.moves.append(self.stage)
        microscope=RotatedMicroscope()
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus',side_effect=focus_result), mock.patch('builtins.print'):
            run=edge.run_particle_edge_workflow(microscope,root,config=edge_fixtures.ParticleEdgeWorkflowTest.config)
            report=json.loads((run/'run.json').read_text())
            baseline=report['particle_view_state']['stage_position_m']
            offsets=[]
            for entry in report['boxes']:
                offset=[entry['box']['offset_x_m'],entry['box']['offset_y_m']]
                offsets.append(offset)
                expected=microscope.get_coordinate_transform().stage_position_for_offset(offset,baseline)
                np.testing.assert_allclose(entry['capture_state']['stage_position_m'],expected,atol=1e-20)
            mosaic=json.loads((run/'particle_0001/mosaic/roi_mosaic.json').read_text())
            np.testing.assert_allclose(mosaic['origin_m'],np.min(offsets,axis=0)-25e-9,atol=1e-20)

    def test_invalid_autofocus_config_is_rejected_before_device_movement(self):
        from dataclasses import replace
        from algorithms.autofocus import AutofocusConfig
        from scripts.usim_particle_workflow import run_particle_workflow, ParticleWorkflowConfig
        microscope=WorkflowMicroscope()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'max_rounds'):
                run_particle_workflow(microscope,root,config=ParticleWorkflowConfig(autofocus=AutofocusConfig(max_rounds=0)))
            self.assertEqual(list(Path(root).iterdir()),[])
            self.assertEqual(microscope.moves,[])
