import csv
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
import numpy as np
from algorithms.FindParticles import PARTICLE_DTYPE
from algorithms.autofocus import AutofocusResult
from adapters.cancellation import WorkflowCancelled
from scripts import thin_particle_workflow as batch
from scripts import usim_thin_workflow as workflow


def particles(count):
    result = np.zeros(count, dtype=PARTICLE_DTYPE)
    result['id'] = np.arange(1, count+1)
    result['offset_x_m'] = np.arange(count)*1e-6
    result['circle_diameter_m'] = 500e-9
    return result


def focus(best=130e-9, converged=True):
    return AutofocusResult(0., best, 1., 5e-9, 4e-9, 3, converged, (), 'precision_reached')


def thin(percent):
    return SimpleNamespace(haadf=np.arange(16).reshape(4, 4), particle_mask=np.ones((4, 4), bool),
        summary=lambda: dict(thin_area_percent=percent, unquantified_area_percent=0.,
                             area_outside_current_fov_percent=0.))


class ThinWorkflowTests(unittest.TestCase):
    def microscope(self):
        microscope = Mock()
        microscope.get_defocus.return_value = 80e-9
        microscope.acquisition_settings.side_effect = lambda **kwargs: nullcontext()
        return microscope

    def test_all_45_particles_use_fixed_reference_relative_focus_and_sorted_ranking(self):
        microscope = self.microscope()
        messages = []
        with tempfile.TemporaryDirectory() as root, \
             patch.object(batch, 'zoom_to_fit', return_value=SimpleNamespace()) as zoom, \
             patch.object(batch, 'asdict', side_effect=lambda value: {} if isinstance(value, SimpleNamespace) else __import__('dataclasses').asdict(value)), \
             patch.object(batch, 'autofocus', side_effect=[focus((100+i)*1e-9) for i in range(45)]), \
             patch.object(batch, 'find_thin', side_effect=[thin(float(i)) for i in range(45)]) as mapping:
            report = batch.map_particles(microscope, particles(45), ['complete']*45, (1e-6, -2e-6),
                                        root, window_report={}, progress=messages.append)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(mapping.call_count, 45)
            self.assertEqual(zoom.call_count, 45)
            for index, call in enumerate(mapping.call_args_list):
                self.assertAlmostEqual(call.kwargs['focus_reference_m'], (100+index)*1e-9)
                self.assertAlmostEqual(call.kwargs['defocus_offset_m'], 0.)
            for call in zoom.call_args_list:
                self.assertEqual(call.kwargs['reference_stage_position_m'], (1e-6, -2e-6))
            self.assertEqual([entry['particle_id'] for entry in report['ranking']], list(range(45, 0, -1)))
            # The second particle starts from the previous true focus, not +200 nm.
            self.assertAlmostEqual(microscope.set_defocus.call_args_list[1].args[0], 100e-9)
            with np.load(Path(root)/'thin_ranking.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['particle_id'], np.arange(45, 0, -1))
            with (Path(root)/'thin_ranking.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['particle_id'], '45')
            self.assertTrue(all('***Warning***' not in message for message in messages))

    def test_cancellation_keeps_completed_ranking(self):
        microscope = self.microscope()
        with tempfile.TemporaryDirectory() as root, \
             patch.object(batch, 'zoom_to_fit', return_value=focus()), \
             patch.object(batch, 'autofocus', return_value=focus()), \
             patch.object(batch, 'find_thin', side_effect=[thin(75.), WorkflowCancelled()]):
            with self.assertRaises(WorkflowCancelled):
                batch.map_particles(microscope, particles(3), ['complete']*3, (0., 0.),
                                    root, window_report={}, progress=lambda message: None)
            report = json.loads((Path(root)/'thin_workflow.json').read_text())
            self.assertEqual(report['status'], 'cancelled')
            self.assertEqual([entry['particle_id'] for entry in report['ranking']], [1])
            self.assertEqual(report['particles'][-1]['status'], 'cancelled')

    def test_unresolved_centers_and_failed_particles_are_recorded_and_excluded(self):
        with tempfile.TemporaryDirectory() as root, \
             patch.object(batch, 'zoom_to_fit', return_value=focus()), \
             patch.object(batch, 'autofocus', return_value=focus(converged=False)), \
             patch.object(batch, 'find_thin', side_effect=[RuntimeError('camera failed'), thin(30.)]):
            report = batch.map_particles(self.microscope(), particles(3), ['unmatched', 'complete', 'recovered'],
                (0., 0.), root, window_report={}, progress=lambda message: None)
            self.assertEqual([entry['status'] for entry in report['particles']],
                             ['skipped_unresolved_center', 'failed', 'mapped_unconverged'])
            self.assertEqual(report['ranking'][0]['particle_id'], 3)
            self.assertFalse(report['ranking'][0]['autofocus_converged'])

    def test_entry_reuses_overview_archive_and_passes_all_particles(self):
        with tempfile.TemporaryDirectory() as root:
            def window(microscope, directory, **kwargs):
                self.assertFalse(kwargs['zoom_first_particle'])
                output = Path(directory)/'cell_0001'
                output.mkdir(parents=True)
                np.savez(output/'find_particles.npz', particles=particles(3), center_status=['complete']*3)
                return {'particle_count': 3, 'cell_reference_stage_xy_m': [1e-6, 2e-6]}
            with patch.object(workflow, 'NionUSimRonchigramAdapter', return_value=MagicMock()), \
                 patch.object(workflow, 'NionUSimEELSAdapter', return_value=MagicMock()), \
                 patch.object(workflow, 'run_window_workflow', side_effect=window), \
                 patch.object(workflow, 'map_particles', return_value={'particles': [], 'status': 'complete'}) as mapping:
                workflow.run_thin_workflow(Mock(), root, progress=lambda message: None)
                self.assertEqual(len(mapping.call_args.args[1]), 3)
                self.assertEqual(mapping.call_args.args[3], [1e-6, 2e-6])

    def test_stage_z_reset_does_not_overwrite_focus(self):
        from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
        adapter = NionUSimRonchigramAdapter(Mock())
        adapter._instrument = Mock()
        adapter._hardware_source = Mock(profile_index=0)
        adapter._get_active_profile_frame_parameters = Mock(return_value={})
        adapter.prepare_window_scan(.4)
        adapter._instrument.set_control_output.assert_called_once_with('stage_z_m', 0.)
