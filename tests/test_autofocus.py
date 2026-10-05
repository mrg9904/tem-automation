import unittest

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
                search_half_range_m=300e-9,
                coarse_points=9,
                fine_points=7,
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
                config=AutofocusConfig(coarse_points=8),
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
                config=AutofocusConfig(search_half_range_m=100e-9),
            )

        self.assertAlmostEqual(microscope.get_defocus(), 90e-9)


if __name__ == "__main__":
    unittest.main()
