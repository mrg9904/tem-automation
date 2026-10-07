import unittest

import numpy as np

from tem_automation.algorithms.autofocus import AutofocusConfig
from tem_automation.algorithms.autofocus import autofocus
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


if __name__ == "__main__":
    unittest.main()
