import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from algorithms.FindParticles import FindParticlesConfig
from algorithms.FindEdge import FindEdgeConfig
from scripts import usim_particle_edge_workflow as workflow
from tests.test_particle_workflow import WorkflowMicroscope, focus_result


class EdgeMicroscope(WorkflowMicroscope):
    def acquire_haadf(self):
        self.acquisitions += 1
        image = np.zeros((80, 80), dtype=np.float32)
        if self.empty:
            return image
        if self.fov > 500e-9:  # Overview #1 is offset from the image center.
            image[10:21, 10:21] = 10
            image[50:61, 50:61] = 10
        else:  # Centered close-up used for edge tracing and final captures.
            image[15:65, 15:65] = 10
        return image


class ParticleEdgeWorkflowTest(unittest.TestCase):
    detection = FindParticlesConfig(gaussian_sigma_px=0, min_area_px=1)
    config = workflow.ParticleEdgeWorkflowConfig(particle_id=1, preserve_last_scan=False, finding=detection,
             edges=FindEdgeConfig(particles=detection))

    def test_real_find_edge_and_every_box_has_50nm_capture_without_accumulation(self):
        microscope = EdgeMicroscope()
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus', side_effect=focus_result), mock.patch('builtins.print'):
            run = workflow.run_particle_edge_workflow(microscope, root, config=self.config)
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertTrue(report['state_restored'])
            self.assertGreater(report['box_count'], 1)
            baseline = np.asarray(report['particle_view_state']['stage_position_m'])
            np.testing.assert_allclose(baseline, [1467e-9, 524e-9])
            with np.load(run / 'particle_0001/find_edge.npz', allow_pickle=False) as edge:
                self.assertEqual(len(edge['boxes']), report['box_count'])
                for entry, box in zip(report['boxes'], edge['boxes']):
                    directory = run / entry['directory']
                    expected = baseline - [box['offset_x_m'], box['offset_y_m']]
                    with np.load(directory / 'haadf.npz', allow_pickle=False) as capture:
                        self.assertAlmostEqual(float(capture['fov_m']), 50e-9, delta=1e-18)
                        self.assertEqual(capture['particle_id'], 1)
                        np.testing.assert_allclose(capture['stage_position_m'], expected)
                    for filename in ('haadf.png', 'autofocus.npz', 'autofocus.json', 'zoom.json', 'box.json'):
                        self.assertTrue((directory / filename).is_file())
            self.assertTrue((run / 'particle_0001/haadf_edge_boxes.png').is_file())
        self.assertEqual(microscope.stage, (1222e-9, 279e-9))
        self.assertEqual(microscope.fov, 800e-9)
        self.assertEqual(microscope.focus, 30e-9)

    def test_failed_box_is_logged_and_later_boxes_continue(self):
        microscope = EdgeMicroscope()
        count = 0
        def fail_first(microscope, config):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError('edge focus failed')
            return focus_result(microscope, config)
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus', side_effect=fail_first), mock.patch('builtins.print'):
            run = workflow.run_particle_edge_workflow(microscope, root, config=self.config)
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['status'], 'completed_with_errors')
            self.assertEqual(report['boxes'][0]['status'], 'error')
            self.assertTrue(all(entry['status'] == 'captured' for entry in report['boxes'][1:]))
            self.assertTrue(report['state_restored'])

    def test_no_particle_aborts_with_report_and_restores(self):
        microscope = EdgeMicroscope(empty=True)
        with tempfile.TemporaryDirectory() as root, mock.patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'No particle'):
                workflow.run_particle_edge_workflow(microscope, root, config=self.config)
            report = json.loads((next(Path(root).iterdir()) / 'run.json').read_text())
            self.assertEqual(report['status'], 'aborted')
            self.assertTrue(report['state_restored'])

    def test_nion_script_runs_without_file_global(self):
        import scripts
        source = Path(workflow.__file__).read_text(encoding='utf-8')
        microscope = EdgeMicroscope()
        with tempfile.TemporaryDirectory() as root, mock.patch.object(scripts, '__file__', str(Path(root) / '__init__.py')), \
             mock.patch('adapters.nion_usim.NionUSimAdapter') as adapter, \
             mock.patch('scripts.common.autofocus', side_effect=focus_result), mock.patch('builtins.print'):
            adapter.return_value.__enter__.return_value = microscope
            original_is_file = Path.is_file
            namespace = {}
            exec(compile(source, 'nion_edge_workflow', 'exec'), namespace)
            with mock.patch.object(namespace['ROIReferenceBank'], 'load', return_value=mock.Mock(threshold=0.1)), mock.patch.object(namespace['Path'], 'is_file', autospec=True, side_effect=lambda path: path.name == 'roi_reference.npz' or original_is_file(path)), mock.patch.dict(namespace, {'run_particle_edge_workflow': lambda microscope, root, **kwargs: workflow.run_particle_edge_workflow(microscope, root, config=self.config)}):
                namespace['script_main'](mock.Mock())
            run = next((Path(root) / 'particle_edge_workflow_results').iterdir())
            report = json.loads((run / 'run.json').read_text())
            self.assertGreater(report['box_count'], 0)
            self.assertTrue(report['state_restored'])

    def test_no_reference_return_between_edge_boxes(self):
        microscope = EdgeMicroscope()
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus', side_effect=focus_result), \
             mock.patch('scripts.common.restore_state', wraps=__import__('scripts.common',fromlist=['restore_state']).restore_state) as restore, mock.patch('builtins.print'):
            run = workflow.run_particle_edge_workflow(microscope, root, config=self.config)
            report = json.loads((run / 'run.json').read_text())
            self.assertGreater(report['box_count'], 1)
            self.assertEqual(restore.call_count, 1)  # Only final restoration to the initial overview.

    def test_particle_two_roi_uses_saved_capture_and_records_locations(self):
        from dataclasses import replace
        from algorithms.FindROI import ROIReferenceBank
        microscope = EdgeMicroscope()
        sample = microscope.acquire_haadf()
        bank = ROIReferenceBank().fit([(sample, 50e-9)])
        bank.calibrate([(sample, 50e-9)])
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus', side_effect=focus_result), mock.patch('builtins.print'):
            run = workflow.run_particle_edge_workflow(microscope, root,
                config=replace(self.config, particle_id=2), roi_reference=bank)
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['particle_id'], 2)
            self.assertEqual(report['mosaic']['status'], 'completed')
            self.assertTrue((run / 'particle_0002/mosaic/haadf_mosaic_roi.png').is_file())
            self.assertEqual(report['roi_reference']['model_id'], bank.model_id)
            self.assertTrue((run / 'particle_0002').is_dir())
            self.assertTrue(any(entry['roi_count'] > 0 for entry in report['boxes']))
            for entry in report['boxes']:
                self.assertEqual(entry['status'], 'captured')
                self.assertEqual(entry['roi_status'], 'completed')
                directory = run / entry['directory']
                self.assertTrue((directory / 'roi/haadf_roi.png').is_file())
                positions = json.loads((directory / 'roi/roi_positions.json').read_text())
                self.assertEqual(positions['particle_id'], 2)
                self.assertEqual(positions['capture_state'], entry['capture_state'])
                with np.load(directory / 'haadf.npz') as capture:
                    self.assertEqual(int(capture['particle_id']), 2)
            np.testing.assert_allclose(report['particle_view_state']['stage_position_m'], [1067e-9,124e-9])

    def test_missing_selected_particle_aborts_before_zoom_and_restores(self):
        from dataclasses import replace
        microscope = EdgeMicroscope()
        config = replace(self.config, particle_id=3)
        with tempfile.TemporaryDirectory() as root, mock.patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'Particle #3 not found'):
                workflow.run_particle_edge_workflow(microscope, root, config=config)
            report = json.loads((next(Path(root).iterdir()) / 'run.json').read_text())
            self.assertEqual(report['status'], 'aborted')
            self.assertTrue(report['state_restored'])
            self.assertEqual(report['box_count'], 0)

    def test_completed_run_preserves_last_capture_and_warns_on_unconverged_focus(self):
        from dataclasses import replace
        microscope = EdgeMicroscope()
        def unconverged(microscope, config):
            return replace(focus_result(microscope, config), converged=False, stop_reason='low_score_confidence')
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.common.autofocus', side_effect=unconverged), mock.patch('builtins.print') as output:
            run = workflow.run_particle_edge_workflow(microscope, root,
                config=replace(self.config,preserve_last_scan=True))
            report = json.loads((run/'run.json').read_text())
            last=report['boxes'][-1]['capture_state']
            self.assertTrue(report['last_scan_preserved'])
            np.testing.assert_allclose(microscope.stage,last['stage_position_m'])
            self.assertAlmostEqual(microscope.fov,last['fov_m'])
            self.assertAlmostEqual(microscope.focus,last['defocus_m'])
            self.assertTrue(all(entry['autofocus_retried'] for entry in report['boxes']))
            self.assertTrue(any(str(call.args[0]).startswith('***Warning***') for call in output.call_args_list))
            self.assertIn('#b65b00',(run/'workflow_messages.html').read_text())
            self.assertTrue((run/report['boxes'][-1]['directory']/'autofocus_initial.json').is_file())
