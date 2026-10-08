import unittest
from unittest import mock

from adapters.cancellation import WorkflowCancelled
from scripts import usim_particle_edge_workflow as workflow
import test_particle_workflow as fixtures


class ParticleEdgeSequenceTest(unittest.TestCase):
    def test_order_overview_restoration_and_final_scan_preserved(self):
        microscope = fixtures.WorkflowMicroscope()
        initial = workflow.InstrumentState.read(microscope)
        starts = []
        reference = object()

        def capture(device, root, *, config, roi_reference):
            self.assertIs(roi_reference, reference)
            starts.append((config.particle_id, workflow.InstrumentState.read(device)))
            device.set_stage_position(config.particle_id * 1e-6, 0)
            device.set_fov(50e-9)
            device.set_defocus(config.particle_id * 1e-9)
            return root / str(config.particle_id)

        from pathlib import Path
        with mock.patch.object(workflow, 'run_particle_edge_workflow', side_effect=capture), mock.patch('builtins.print'):
            paths = workflow.run_particle_edge_sequence(microscope, Path('results'), roi_reference=reference)
        self.assertEqual([identifier for identifier, _ in starts], [4, 5, 6])
        self.assertTrue(all(state == initial for _, state in starts))
        self.assertEqual([p.name for p in paths], ['4', '5', '6'])
        self.assertEqual(microscope.stage, (6e-6, 0))
        self.assertEqual(microscope.fov, 50e-9)

    def test_cancel_stops_remaining_particles(self):
        microscope = fixtures.WorkflowMicroscope()
        with mock.patch.object(workflow, 'run_particle_edge_workflow', side_effect=WorkflowCancelled()) as capture:
            with self.assertRaises(WorkflowCancelled):
                workflow.run_particle_edge_sequence(microscope, 'results')
        self.assertEqual(capture.call_count, 1)

    def test_invalid_ids_rejected_before_first_particle(self):
        microscope = fixtures.WorkflowMicroscope()
        with mock.patch.object(workflow, 'run_particle_edge_workflow') as capture:
            with self.assertRaises(ValueError):
                workflow.run_particle_edge_sequence(microscope, 'results', particle_ids=(4, 0, 6))
        capture.assert_not_called()
