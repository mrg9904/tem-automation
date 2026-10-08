import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from adapters.cancellation import WorkflowCancelled, cancellation_scope, check_cancelled, set_stop_file
from adapters.nion_usim import NionUSimAdapter
from algorithms.autofocus import autofocus
from scripts.usim_particle_edge_workflow import run_particle_edge_workflow
from tests import test_particle_edge_workflow as edge_fixtures
from tests.test_nion_usim_adapter import _FakeAPI
from tests.fake_microscope import FakeMicroscope


class CancellationTest(unittest.TestCase):
    def test_cancel_during_record_aborts_without_waiting_for_record_timeout(self):
        api = _FakeAPI()
        started, stopped = threading.Event(), threading.Event()
        def record(*args):
            started.set()
            stopped.wait(2)
            return []
        api.hardware_source.record = record
        api.hardware_source.abort_recording = stopped.set
        with NionUSimAdapter(api) as microscope, cancellation_scope(started.is_set):
            with self.assertRaises(WorkflowCancelled):
                microscope.acquire_haadf()
        self.assertTrue(stopped.is_set())

    def test_cancel_autofocus_restores_original_focus(self):
        microscope = FakeMicroscope(initial_defocus_m=30e-9)
        original = microscope.acquire_haadf
        cancelled = threading.Event()
        def acquire():
            cancelled.set()
            return original()
        microscope.acquire_haadf = acquire
        with cancellation_scope(cancelled.is_set), self.assertRaises(WorkflowCancelled):
            autofocus(microscope)
        self.assertEqual(microscope.get_defocus(), 30e-9)

    def test_cancel_workflow_is_not_swallowed_by_continue_on_error(self):
        microscope = edge_fixtures.EdgeMicroscope()
        with tempfile.TemporaryDirectory() as root, mock.patch('scripts.usim_particle_workflow.autofocus', side_effect=WorkflowCancelled('stop')), mock.patch('builtins.print'):
            with self.assertRaises(WorkflowCancelled):
                run_particle_edge_workflow(microscope, root, config=edge_fixtures.ParticleEdgeWorkflowTest.config)
            report = json.loads((next(Path(root).iterdir()) / 'run.json').read_text())
            self.assertEqual(report['status'], 'cancelled')
            self.assertTrue(report['state_restored'])
            self.assertEqual(len(report['boxes']), 1)
        self.assertEqual(microscope.get_defocus(), 30e-9)
        self.assertEqual(microscope.stage, (1222e-9, 279e-9))

    def test_stop_file_is_an_independent_stop_mechanism(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'STOP'
            previous = set_stop_file(path)
            try:
                check_cancelled()
                path.touch()
                with self.assertRaises(WorkflowCancelled):
                    check_cancelled()
            finally:
                set_stop_file(previous)
