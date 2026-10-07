from pathlib import Path
import tempfile
import unittest
from unittest import mock
from PIL import Image
from scripts import usim_find_particles

import numpy as np

from algorithms.FindParticles import (
    FindParticlesConfig, find_particles, find_particles_in_image,
)


class FindParticlesTest(unittest.TestCase):
    def test_centers_ids_and_rectangular_image_coordinates(self):
        image = np.zeros((40, 80), dtype=np.float32)
        image[5:10, 10:15] = 10
        image[25:30, 50:55] = 10
        image[20, 20] = 10  # Noise is below the area cutoff.
        result = find_particles_in_image(image, fov_m=800e-9,
            config=FindParticlesConfig(gaussian_sigma_px=0, min_area_px=4))
        np.testing.assert_array_equal(result.particles['id'], [1, 2])
        np.testing.assert_allclose(result.particles['center_x_px'], [12, 52])
        np.testing.assert_allclose(result.particles['center_y_px'], [7, 27])
        np.testing.assert_allclose(result.particles['offset_x_m'], [-275e-9, 125e-9])
        np.testing.assert_allclose(result.particles['offset_y_m'], [-125e-9, 75e-9])
        np.testing.assert_array_equal(result.particles['area_px'], [25, 25])
        self.assertEqual(result.labels[7, 12], 1)
        self.assertEqual(result.labels[27, 52], 2)
        self.assertEqual(result.labels[20, 20], 0)

    def test_noisy_bright_particles(self):
        image = np.random.default_rng(17).normal(0, 0.2, (80, 80))
        y, x = np.mgrid[:80, :80]
        image[(x - 20)**2 + (y - 20)**2 <= 8**2] += 5
        image[(x - 55)**2 + (y - 55)**2 <= 10**2] += 5
        result = find_particles_in_image(image, fov_m=80e-9)
        self.assertEqual(len(result.particles), 2)
        np.testing.assert_allclose(result.particles['center_x_px'], [20, 55], atol=0.5)
        np.testing.assert_allclose(result.particles['center_y_px'], [20, 55], atol=0.5)

    def test_edges_are_flagged_and_optionally_excluded(self):
        image = np.zeros((20, 20))
        image[:5, :5] = 3
        image[10:15, 10:15] = 3
        for exclude, expected in ((False, 2), (True, 1)):
            result = find_particles_in_image(image, fov_m=20e-9,
                config=FindParticlesConfig(gaussian_sigma_px=0, exclude_edge_particles=exclude))
            self.assertEqual(len(result.particles), expected)
            np.testing.assert_array_equal(result.particles['touches_edge'], [True, False] if not exclude else [False])
            np.testing.assert_array_equal(result.particles['id'], np.arange(1, expected + 1))

    def test_uniform_images_have_no_particles(self):
        for bright in (True, False):
            result = find_particles_in_image(np.ones((20, 20)), fov_m=1e-6,
                config=FindParticlesConfig(bright_particles=bright))
            self.assertEqual(result.particles.shape, (0,))
            self.assertFalse(result.labels.any())

    def test_explicit_threshold_dark_particles_and_connectivity(self):
        image = np.ones((20, 20)) * 10
        image[2:5, 2:5] = 0
        image[5:8, 5:8] = 0  # Diagonal contact joins the two regions.
        result = find_particles_in_image(image, fov_m=1e-6,
            config=FindParticlesConfig(gaussian_sigma_px=0, min_area_px=1,
                                        threshold=5, bright_particles=False))
        self.assertEqual(len(result.particles), 1)
        self.assertEqual(result.particles['area_px'][0], 18)

    def test_archive_roundtrip_and_acquisition_preserves_focus(self):
        class Microscope:
            focus = 20e-9
            acquisitions = 0
            def get_fov(self):
                return 100e-9
            def acquire_haadf(self):
                self.acquisitions += 1
                image = np.zeros((40, 40), dtype=np.float32)
                image[10:20, 10:20] = 10
                return image
        microscope = Microscope()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'particles.npz'
            result = find_particles(microscope, output_path=path)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['particles'], result.particles)
                np.testing.assert_array_equal(archive['labels'], result.labels)
                np.testing.assert_array_equal(archive['image'], result.image)
                self.assertEqual(archive['schema_version'], 2)
                self.assertEqual(archive['fov_m'], 100e-9)
        self.assertEqual(microscope.acquisitions, 1)
        self.assertEqual(microscope.focus, 20e-9)

    def test_invalid_inputs_are_rejected(self):
        for image in (np.zeros((0, 2)), np.zeros((2, 2, 2)), [[np.nan]], [[np.inf]], [[1j]]):
            with self.subTest(image=image), self.assertRaises(ValueError):
                find_particles_in_image(image, fov_m=1e-6)
        for fov in (0, -1, np.inf, np.nan):
            with self.subTest(fov=fov), self.assertRaises(ValueError):
                find_particles_in_image(np.zeros((2, 2)), fov_m=fov)
        for config in (FindParticlesConfig(min_area_px=0), FindParticlesConfig(min_area_px=1.5),
                       FindParticlesConfig(gaussian_sigma_px=-1), FindParticlesConfig(threshold=np.nan)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                find_particles_in_image(np.zeros((2, 2)), fov_m=1e-6, config=config)

    def test_saved_images_overlay_matches_centers_and_keeps_raw_data(self):
        image = np.zeros((80, 80), dtype=np.float32)
        image[30:41, 30:41] = 10
        image[:6, 60:70] = 10
        result = find_particles_in_image(image, fov_m=80e-9,
            config=FindParticlesConfig(gaussian_sigma_px=0))
        with tempfile.TemporaryDirectory() as directory:
            raw_path, annotated_path = result.save_images(directory)
            with Image.open(raw_path) as raw, Image.open(annotated_path) as annotated:
                self.assertEqual(raw.size, (80, 80))
                self.assertEqual(annotated.size, (80, 80))
                self.assertEqual(raw.getpixel((35, 35)), 255)
                self.assertEqual(annotated.getpixel((35, 35)), (0, 255, 255))
                colors = np.asarray(annotated)
                for color in ((0, 255, 0), (0, 255, 255), (255, 255, 0)):
                    self.assertTrue(np.any(np.all(colors == color, axis=2)))
        np.testing.assert_array_equal(result.image, image)

    def test_empty_detection_still_exports_images(self):
        result = find_particles_in_image(np.ones((20, 20)), fov_m=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            raw_path, annotated_path = result.save_images(directory)
            with Image.open(raw_path) as raw, Image.open(annotated_path) as annotated:
                np.testing.assert_array_equal(np.asarray(raw.convert('RGB')), np.asarray(annotated))

    def test_script_saves_complete_bundles_beside_script_without_overwriting(self):
        image = np.zeros((80, 80), dtype=np.float32)
        image[30:41, 30:41] = 10
        microscope = mock.Mock()
        microscope.get_fov.return_value = 80e-9
        microscope.acquire_haadf.return_value = image
        broker = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / 'usim_find_particles.py'
            with mock.patch.object(usim_find_particles, 'scripts_package_file', str(script_path)), \
                 mock.patch.object(usim_find_particles, 'NionUSimAdapter') as adapter, \
                 mock.patch('builtins.print'):
                adapter.return_value.__enter__.return_value = microscope
                usim_find_particles.script_main(broker)
                usim_find_particles.script_main(broker)
            runs = list((Path(directory) / 'find_particles_results').iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertEqual({p.name for p in run.iterdir()},
                    {'find_particles.npz', 'haadf.png', 'haadf_annotated.png'})
                with np.load(run / 'find_particles.npz', allow_pickle=False) as data:
                    self.assertEqual(len(data['particles']), 1)
                    np.testing.assert_array_equal(data['image'], image)
            self.assertEqual(microscope.acquire_haadf.call_count, 2)

    def test_nion_exec_without_file_global_saves_under_scripts_package(self):
        import scripts
        source = Path(usim_find_particles.__file__).read_text(encoding='utf-8')
        microscope = mock.Mock()
        microscope.get_fov.return_value = 80e-9
        microscope.acquire_haadf.return_value = np.zeros((80, 80), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(scripts, '__file__', str(Path(directory) / '__init__.py')), \
                 mock.patch('adapters.nion_usim.NionUSimAdapter') as adapter, \
                 mock.patch('builtins.print'):
                adapter.return_value.__enter__.return_value = microscope
                namespace = {}  # Nion Swift executes scripts without __file__.
                exec(compile(source, 'nion_script', 'exec'), namespace)
                namespace['script_main'](mock.Mock())
            runs = list((Path(directory) / 'find_particles_results').iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / 'haadf_annotated.png').is_file())

    def test_circle_diameters_match_enclosing_circle_and_physical_scale(self):
        image = np.zeros((40, 80), dtype=np.float32)
        image[5:10, 10:15] = 10
        image[20:23, 50:53] = 10
        result = find_particles_in_image(image, fov_m=800e-9,
            config=FindParticlesConfig(gaussian_sigma_px=0, min_area_px=1))
        # Small detected regions use the same minimum 4-pixel radius as the overlay.
        np.testing.assert_allclose(result.particles['circle_diameter_px'], [8, 8])
        np.testing.assert_allclose(result.particles['circle_diameter_m'], [80e-9, 80e-9])
        image[:] = 0
        image[10:21, 10:21] = 10
        result = find_particles_in_image(image, fov_m=800e-9,
            config=FindParticlesConfig(gaussian_sigma_px=0, min_area_px=1))
        expected = 2 * (np.hypot(5, 5) + 1)
        self.assertAlmostEqual(result.particles['circle_diameter_px'][0], expected)
        self.assertAlmostEqual(result.particles['circle_diameter_m'][0], expected * 10e-9, delta=1e-18)
        with tempfile.TemporaryDirectory() as directory:
            path = result.save(Path(directory) / 'particles.npz')
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['particles']['circle_diameter_px'], result.particles['circle_diameter_px'])
                np.testing.assert_array_equal(data['particles']['circle_diameter_m'], result.particles['circle_diameter_m'])
