import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from algorithms.FindParticles import FindParticlesConfig
from algorithms.autofocus import AutofocusConfig, AutofocusResult, FocusMeasurement
from scripts import usim_particle_workflow as workflow


class WorkflowMicroscope:
    def __init__(self, empty=False):
        self.stage = (1222e-9, 279e-9)
        self.fov = 800e-9
        self.focus = 30e-9
        self.acquisitions = 0
        self.moves = []
        self.empty = empty

    def get_stage_position(self):
        return self.stage

    def set_stage_position(self, x, y):
        self.stage = (x, y)

    def get_fov(self):
        return self.fov

    def set_fov(self, fov):
        self.fov = fov

    def get_defocus(self):
        return self.focus

    def set_defocus(self, focus):
        self.focus = focus

    def center_fov_on_image_offset(self, x, y):
        self.stage = (self.stage[0] - x, self.stage[1] - y)
        self.moves.append(self.stage)

    def acquire_haadf(self):
        self.acquisitions += 1
        image = np.zeros((80, 80), dtype=np.float32)
        if not self.empty:
            image[10:21, 10:21] = 10
            image[50:61, 50:61] = 10
        return image


def focus_result(microscope, config=None, converged=True):
    original = microscope.focus
    microscope.set_defocus(5e-9)
    return AutofocusResult(original, 5e-9, 1.0, 1e-9, 1e-9, 2, converged,
                           (FocusMeasurement(5e-9, 1.0),))


class ParticleWorkflowTest(unittest.TestCase):
    config = workflow.ParticleWorkflowConfig(finding=FindParticlesConfig(gaussian_sigma_px=0))

    def test_all_particles_capture_correct_absolute_positions_and_restore_state(self):
        microscope = WorkflowMicroscope()
        with tempfile.TemporaryDirectory() as root, mock.patch.object(workflow, 'autofocus', side_effect=focus_result), mock.patch('builtins.print'):
            run = workflow.run_particle_workflow(microscope, root, config=self.config)
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['particle_count'], 2)
            self.assertTrue(report['state_restored'])
            np.testing.assert_allclose(microscope.moves, [(1467e-9, 524e-9), (1067e-9, 124e-9)])
            for index in (1, 2):
                folder = run / f'particle_{index:04d}'
                for name in ('haadf.npz', 'haadf.png', 'zoom.json', 'autofocus.json', 'autofocus.npz', 'particle.json'):
                    self.assertTrue((folder / name).is_file())
                with np.load(folder / 'haadf.npz', allow_pickle=False) as capture:
                    self.assertEqual(capture['particle_id'], index)
                    self.assertEqual(capture['defocus_m'], 5e-9)
                    np.testing.assert_allclose(capture['stage_position_m'], microscope.moves[index - 1])
            self.assertTrue((run / 'overview' / 'haadf_annotated.png').is_file())
        self.assertEqual(microscope.stage, (1222e-9, 279e-9))
        self.assertEqual(microscope.fov, 800e-9)
        self.assertEqual(microscope.focus, 30e-9)
        self.assertEqual(microscope.acquisitions, 3)

    def test_particle_failure_is_logged_and_next_particle_is_processed(self):
        microscope = WorkflowMicroscope()
        calls = 0
        def fail_first(microscope, config):
            nonlocal calls
            calls += 1
            if calls == 1:
                microscope.set_defocus(-50e-9)
                raise RuntimeError('focus failed')
            return focus_result(microscope, config)
        with tempfile.TemporaryDirectory() as root, mock.patch.object(workflow, 'autofocus', side_effect=fail_first), mock.patch('builtins.print'):
            run = workflow.run_particle_workflow(microscope, root, config=self.config)
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['status'], 'completed_with_errors')
            self.assertEqual([p['status'] for p in report['particles']], ['error', 'captured'])
            self.assertIn('focus failed', report['particles'][0]['error'])
            self.assertTrue((run / 'particle_0002' / 'haadf.png').is_file())
        self.assertEqual(microscope.focus, 30e-9)

    def test_fail_fast_restores_and_checkpoints_aborted_run(self):
        microscope = WorkflowMicroscope()
        config = workflow.ParticleWorkflowConfig(continue_on_error=False, finding=self.config.finding)
        with tempfile.TemporaryDirectory() as root, mock.patch.object(workflow, 'autofocus', side_effect=RuntimeError('focus failed')), mock.patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'focus failed'):
                workflow.run_particle_workflow(microscope, root, config=config)
            run = next(Path(root).iterdir())
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['status'], 'aborted')
            self.assertTrue(report['state_restored'])
            self.assertEqual(len(report['particles']), 1)
        self.assertEqual(microscope.stage, (1222e-9, 279e-9))

    def test_empty_overview_saves_report_without_autofocus(self):
        microscope = WorkflowMicroscope(empty=True)
        with tempfile.TemporaryDirectory() as root, mock.patch.object(workflow, 'autofocus') as focus, mock.patch('builtins.print'):
            run = workflow.run_particle_workflow(microscope, root, config=self.config)
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['particle_count'], 0)
            self.assertEqual(report['status'], 'completed')
            focus.assert_not_called()
        self.assertEqual(microscope.acquisitions, 1)

    def test_real_autofocus_records_unconverged_capture(self):
        microscope = WorkflowMicroscope()
        config = workflow.ParticleWorkflowConfig(finding=self.config.finding,
                                                  autofocus=AutofocusConfig(max_rounds=1))
        with tempfile.TemporaryDirectory() as root, mock.patch('builtins.print'):
            run = workflow.run_particle_workflow(microscope, root, config=config)
            report = json.loads((run / 'run.json').read_text())
            self.assertEqual(report['status'], 'completed_with_unconverged_focus')
            self.assertEqual([p['status'] for p in report['particles']], ['captured_unconverged'] * 2)
            with np.load(run / 'particle_0001' / 'autofocus.npz', allow_pickle=False) as focus:
                self.assertEqual(len(focus['defocus_m']), 7)
                self.assertFalse(focus['converged'])
        self.assertEqual(microscope.focus, 30e-9)

    def test_restoration_failure_aborts_and_is_recorded(self):
        microscope = WorkflowMicroscope(empty=True)
        microscope.set_stage_position = mock.Mock(side_effect=RuntimeError('stage unavailable'))
        with tempfile.TemporaryDirectory() as root, mock.patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'Could not restore'):
                workflow.run_particle_workflow(microscope, root, config=self.config)
            report = json.loads((next(Path(root).iterdir()) / 'run.json').read_text())
            self.assertEqual(report['status'], 'restoration_failed')
            self.assertFalse(report['state_restored'])

    def test_nion_entry_point_without_file_global(self):
        import scripts
        source = Path(workflow.__file__).read_text(encoding='utf-8')
        microscope = WorkflowMicroscope(empty=True)
        with tempfile.TemporaryDirectory() as root, mock.patch.object(scripts, '__file__', str(Path(root) / '__init__.py')), \
             mock.patch('adapters.nion_usim.NionUSimAdapter') as adapter, mock.patch('builtins.print'):
            adapter.return_value.__enter__.return_value = microscope
            namespace = {}
            exec(compile(source, 'nion_workflow', 'exec'), namespace)
            namespace['script_main'](mock.Mock())
            report = json.loads((next((Path(root) / 'particle_workflow_results').iterdir()) / 'run.json').read_text())
            self.assertEqual(report['status'], 'completed')
