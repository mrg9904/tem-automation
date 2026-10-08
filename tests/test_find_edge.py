from pathlib import Path
import tempfile
import unittest
import numpy as np
from algorithms.FindParticles import FindParticlesConfig
from algorithms.FindEdge import FindEdgeConfig, find_edge_in_image


class FindEdgeTest(unittest.TestCase):
    config = FindEdgeConfig(particles=FindParticlesConfig(gaussian_sigma_px=0, min_area_px=1), center_on_edge=False)

    def test_closed_edge_exact_size_coverage_and_no_overlap(self):
        image = np.zeros((100, 100))
        image[25:75, 25:75] = 10
        result = find_edge_in_image(image, fov_m=100e-9, box_fov_m=10e-9, config=self.config)
        self.assertEqual(len(result.contour_particle_ids), 1)
        np.testing.assert_array_equal(result.edge_points_px[0], result.edge_points_px[-1])
        self.assertFalse(result.contour_touches_frame.any())
        np.testing.assert_allclose(result.boxes['fov_m'], 10e-9)
        centers = np.column_stack((result.boxes['center_x_px'], result.boxes['center_y_px']))
        covered = np.any(np.max(np.abs(result.edge_points_px[:, None] - centers), axis=2) <= 5 + 1e-9, axis=1)
        self.assertTrue(covered.all())
        distances = np.abs(centers[:, None] - centers)
        overlap = np.all(distances < 10 - 1e-9, axis=2)
        np.fill_diagonal(overlap, False)
        self.assertFalse(overlap.any())
        np.testing.assert_allclose(result.box_centers_m, (centers + .5 - 50) * 1e-9)
        with tempfile.TemporaryDirectory() as directory:
            path = result.save(Path(directory) / 'edge.npz')
            result.save_overlay(Path(directory) / 'edge.png')
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['boxes'], result.boxes)
                np.testing.assert_array_equal(data['edge_points_px'], result.edge_points_px)

    def test_multiple_particles_do_not_bridge_contours(self):
        image = np.zeros((100, 100))
        image[5:20, 5:20] = 10
        image[75:90, 75:90] = 10
        result = find_edge_in_image(image, fov_m=100e-9, box_fov_m=10e-9, config=self.config)
        self.assertEqual(len(result.contour_particle_ids), 2)
        middle = (result.boxes['center_x_px'] > 30) & (result.boxes['center_x_px'] < 65)
        self.assertFalse(middle.any())
        selected = find_edge_in_image(image, fov_m=100e-9, box_fov_m=10e-9, particle_id=2, config=self.config)
        np.testing.assert_array_equal(selected.contour_particle_ids, [2])

    def test_holes_and_frame_edges_are_reported(self):
        image = np.zeros((60, 60))
        image[:50, :50] = 10
        image[15:25, 15:25] = 0
        outer = find_edge_in_image(image, fov_m=60e-9, box_fov_m=10e-9, config=self.config)
        self.assertEqual(len(outer.contour_offsets), 2)
        self.assertTrue(outer.contour_touches_frame.all())
        all_edges = find_edge_in_image(image, fov_m=60e-9, box_fov_m=10e-9,
            config=FindEdgeConfig(particles=self.config.particles, outer_edges_only=False))
        self.assertEqual(len(all_edges.contour_particle_ids), 2)

    def test_empty_and_invalid_inputs(self):
        result = find_edge_in_image(np.zeros((20, 20)), fov_m=20e-9, box_fov_m=10e-9, config=self.config)
        self.assertEqual(result.box_centers_m.shape, (0, 2))
        for size in (0, -1, np.nan, .1e-9):
            with self.subTest(size=size), self.assertRaises(ValueError):
                find_edge_in_image(np.zeros((20, 20)), fov_m=20e-9, box_fov_m=size, config=self.config)

    def test_edge_centered_boxes_cover_edge_with_centers_on_boundary(self):
        image = np.zeros((100, 100))
        image[25:75, 25:75] = 10
        result = find_edge_in_image(image, fov_m=100e-9, box_fov_m=10e-9,
            config=FindEdgeConfig(particles=self.config.particles))
        centers = np.column_stack((result.boxes['center_x_px'], result.boxes['center_y_px']))
        distance = np.max(np.abs(result.edge_points_px[:, None] - centers), axis=2)
        self.assertTrue(np.all(np.any(distance <= 5 + 1e-9, axis=1)))
        self.assertTrue(np.all(np.isin(centers[:, 0], [24.5, 74.5]) | np.isin(centers[:, 1], [24.5, 74.5])))

    def test_sequence_follows_closed_contour_instead_of_alternating_sides(self):
        image = np.zeros((100, 100))
        image[20:80, 20:80] = 10
        result = find_edge_in_image(image, fov_m=100e-9, box_fov_m=10e-9,
            config=FindEdgeConfig(particles=self.config.particles))
        np.testing.assert_array_equal(result.boxes['id'], np.arange(1, len(result.boxes)+1))
        self.assertTrue(np.all(np.diff(result.boxes['arc_length_px']) >= 0))
        centers = np.column_stack((result.boxes['center_x_px'], result.boxes['center_y_px']))
        distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        self.assertLess(distances.max(), 20)
        # The sequence passes the bottom edge before climbing the left side.
        right = np.where(centers[:,0] == 79.5)[0]
        left = np.where(centers[:,0] == 19.5)[0]
        self.assertLess(right.min(), left.max())
