import unittest

import numpy as np

from algorithms.autofocus import AutofocusConfig
from algorithms.autofocus import autofocus
from tests.fake_microscope import FakeMicroscope


class AutofocusTest(unittest.TestCase):
    def test_autofocus_finds_known_best_focus(self) -> None:
        microscope = FakeMicroscope(
            initial_defocus_m=-120e-9,
            best_defocus_m=65e-9,
        )
        microscope.connect()

        result = autofocus(
            microscope,
            config=AutofocusConfig(
                initial_half_range_fov_fraction=3.0,
                points_per_round=9,
            ),
        )

        self.assertAlmostEqual(
            result.best_defocus_m,
            65e-9,
            delta=30e-9,
        )
        self.assertEqual(
            microscope.get_defocus(),
            result.best_defocus_m,
        )
        self.assertGreaterEqual(len(result.measurements), 13)

    def test_autofocus_rejects_even_number_of_points(self) -> None:
        microscope = FakeMicroscope()
        microscope.connect()

        with self.assertRaises(ValueError):
            autofocus(
                microscope,
                config=AutofocusConfig(points_per_round=8),
            )

    def test_autofocus_restores_original_focus_after_failure(self) -> None:
        class FailingMicroscope(FakeMicroscope):
            def acquire_haadf(self):
                raise RuntimeError("simulated acquisition failure")

        microscope = FailingMicroscope(initial_defocus_m=90e-9)
        microscope.connect()

        with self.assertRaises(RuntimeError):
            autofocus(
                microscope,
                config=AutofocusConfig(initial_half_range_fov_fraction=1.0),
            )

        self.assertAlmostEqual(microscope.get_defocus(), 90e-9)

    def test_initial_total_range_is_twice_fov(self) -> None:
        microscope = FakeMicroscope(initial_defocus_m=20e-9)
        result = autofocus(microscope, config=AutofocusConfig(max_rounds=1))
        positions = [item.defocus_m for item in result.measurements]
        self.assertAlmostEqual(min(positions), -80e-9, delta=1e-18)
        self.assertAlmostEqual(max(positions), 120e-9, delta=1e-18)
        self.assertAlmostEqual(max(positions) - min(positions), 2 * microscope.get_fov(), delta=1e-18)

    def test_expands_range_to_bracket_peak_in_either_direction(self) -> None:
        class PeakMicroscope(FakeMicroscope):
            def acquire_haadf(self):
                error = (self.defocus_m - self.best_defocus_m) / self.get_fov()
                return np.full((2, 2), -error ** 2, dtype=np.float32)

        for sign in (-1, 1):
            with self.subTest(sign=sign):
                microscope = PeakMicroscope(initial_defocus_m=0., best_defocus_m=sign * 1000e-9)
                result = autofocus(microscope, metric=lambda image: float(image.mean()))
                self.assertTrue(result.converged)
                self.assertAlmostEqual(result.best_defocus_m, microscope.best_defocus_m,
                                       delta=result.target_precision_m)
                positions = [item.defocus_m / microscope.get_fov() for item in result.measurements]
                np.testing.assert_allclose(positions[:7], np.linspace(-1, 1, 7))
                # The next round is centered on the winning endpoint, +/- 2 FoV.
                self.assertTrue(any(abs(position - sign * 3) < 1e-8 for position in positions[7:]))

    def test_unbracketed_peak_exhausts_budget_without_claiming_convergence(self) -> None:
        microscope = FakeMicroscope(initial_defocus_m=0.)
        result = autofocus(microscope, config=AutofocusConfig(max_rounds=2),
                           metric=lambda image: microscope.get_defocus())
        self.assertFalse(result.converged)
        self.assertEqual(result.rounds, 2)
        self.assertAlmostEqual(result.best_defocus_m, 3 * microscope.get_fov(), delta=1e-18)

    def test_flat_scores_stop_without_expansion_or_focus_drift(self):
        microscope = FakeMicroscope(initial_defocus_m=30e-9)
        result = autofocus(microscope, metric=lambda image: 1.0)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.stop_reason, 'score_plateau')
        self.assertAlmostEqual(result.best_defocus_m, 30e-9, delta=1e-18)
        self.assertFalse(result.converged)

    def test_monotone_metric_has_a_bounded_search(self):
        microscope = FakeMicroscope(initial_defocus_m=0.)
        result = autofocus(microscope, config=AutofocusConfig(max_expansion_rounds=1),
                           metric=lambda image: microscope.get_defocus())
        self.assertEqual(result.stop_reason, 'expansion_limit')
        self.assertEqual(result.rounds, 2)
        self.assertFalse(result.converged)
        self.assertLessEqual(max(abs(m.defocus_m) for m in result.measurements), 3.01 * microscope.get_fov())

    def test_bracketed_peak_never_reexpands_on_later_endpoint_scores(self):
        microscope = FakeMicroscope(initial_defocus_m=0.)
        count = 0
        def noisy_metric(image):
            nonlocal count
            count += 1
            x = microscope.get_defocus() / microscope.get_fov()
            return -x*x if count <= 7 else x + 2
        result = autofocus(microscope,
            config=AutofocusConfig(frames_per_position=1, score_relative_tolerance=0), metric=noisy_metric)
        self.assertLessEqual(max(abs(m.defocus_m) for m in result.measurements[7:]), microscope.get_fov() / 3 + 1e-18)

    def test_edge_focus_near_130nm_converges_with_practical_precision(self):
        microscope = FakeMicroscope(initial_defocus_m=0.)
        calls = 0
        def peak(image):
            nonlocal calls
            calls += 1
            return 1.0 - ((microscope.get_defocus() - 130e-9) / 200e-9)**2
        result = autofocus(microscope, metric=peak,
            config=AutofocusConfig(initial_half_range_fov_fraction=2, points_per_round=5,
                frames_per_position=1, minimum_precision_m=5e-9))
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.best_defocus_m, 130e-9, delta=5e-9)
        self.assertLessEqual(calls, 25)

    def test_small_noisy_interior_peak_is_not_reported_as_converged(self):
        microscope = FakeMicroscope(initial_defocus_m=140e-9)
        values = iter([1.,1.006,1.004,1.003,1.002])
        result = autofocus(microscope, config=AutofocusConfig(points_per_round=5,
            frames_per_position=1, minimum_score_span_fraction=.02),
            metric=lambda image: next(values))
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason,'low_score_confidence')
        self.assertEqual(result.rounds,1)


if __name__ == "__main__":
    unittest.main()
