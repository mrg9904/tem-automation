import unittest
from unittest import mock

import numpy as np

from algorithms.Zoom2Fit import zoom_to_fit


class FakePositioningMicroscope:
    def __init__(self):
        self.fov = 10000e-9
        self.stage = (1222e-9, 279e-9)
        self.fail_next_fov = False

    def get_fov(self):
        return self.fov

    def set_fov(self, value):
        self.fov = value
        if self.fail_next_fov:
            self.fail_next_fov = False
            raise RuntimeError('simulated FoV failure')

    def get_stage_position(self):
        return self.stage

    def set_stage_position(self, x, y):
        self.stage = (x, y)

    def center_fov_on_image_offset(self, x, y):
        self.stage = (self.stage[0] - x, self.stage[1] - y)


class Zoom2FitTest(unittest.TestCase):
    def test_center_and_padding(self):
        microscope = FakePositioningMicroscope()
        result = zoom_to_fit(microscope, (100e-9, -50e-9), 200e-9, padding=1.1)
        np.testing.assert_allclose(result.actual_stage_position_m, (1122e-9, 329e-9))
        self.assertAlmostEqual(result.actual_fov_m, 220e-9, delta=1e-18)
        self.assertEqual(result.original_stage_position_m, (1222e-9, 279e-9))

    def test_failure_restores_stage_and_fov(self):
        microscope = FakePositioningMicroscope()
        microscope.fail_next_fov = True
        with self.assertRaisesRegex(RuntimeError, 'simulated FoV failure'):
            zoom_to_fit(microscope, (100e-9, 50e-9), 200e-9)
        self.assertEqual(microscope.stage, (1222e-9, 279e-9))
        self.assertEqual(microscope.fov, 10000e-9)

    def test_partial_stage_failure_also_restores_state(self):
        microscope = FakePositioningMicroscope()
        def partial_move(x, y):
            microscope.stage = (0.0, microscope.stage[1])
            raise RuntimeError('stage failed')
        microscope.center_fov_on_image_offset = partial_move
        with self.assertRaisesRegex(RuntimeError, 'stage failed'):
            zoom_to_fit(microscope, (100e-9, 50e-9), 200e-9)
        self.assertEqual(microscope.stage, (1222e-9, 279e-9))
        self.assertEqual(microscope.fov, 10000e-9)

    def test_incomplete_restoration_is_reported_and_attempts_both(self):
        microscope = FakePositioningMicroscope()
        microscope.set_fov = mock.Mock(side_effect=RuntimeError('FoV unavailable'))
        with self.assertRaisesRegex(RuntimeError, 'restoration was incomplete'):
            zoom_to_fit(microscope, (100e-9, 50e-9), 200e-9)
        self.assertEqual(microscope.stage, (1222e-9, 279e-9))

    def test_invalid_inputs_do_not_move_instrument(self):
        for center, diameter, padding in (((np.nan, 0), 1e-6, 1), ((0,), 1e-6, 1),
            ((0, 0), 0, 1), ((0, 0), np.inf, 1), ((0, 0), 1e-6, 0.9)):
            microscope = mock.Mock()
            with self.subTest(center=center, diameter=diameter, padding=padding), self.assertRaises(ValueError):
                zoom_to_fit(microscope, center, diameter, padding=padding)
            self.assertEqual(microscope.mock_calls, [])

    def test_returns_actual_fov_when_instrument_rounds_size(self):
        microscope = FakePositioningMicroscope()
        microscope.set_fov = lambda value: setattr(microscope, 'fov', round(value * 1e9) * 1e-9)
        result = zoom_to_fit(microscope, (0, 0), 200.4e-9)
        self.assertAlmostEqual(result.actual_fov_m, 200e-9, delta=1e-18)
        self.assertEqual(result.requested_fov_m, 200.4e-9)
