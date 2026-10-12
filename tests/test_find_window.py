import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from algorithms.FindWindow import find_window_in_image, make_window_anchor, subdivide_window


def square_image(angle, center=(130., 122.), side=112., invert=False):
    y, x = np.mgrid[:256, :256]
    theta = np.radians(angle)
    u = (x-center[0])*np.cos(theta)+(y-center[1])*np.sin(theta)
    v = -(x-center[0])*np.sin(theta)+(y-center[1])*np.cos(theta)
    mask = (abs(u) <= side/2) & (abs(v) <= side/2)
    rng = np.random.default_rng(7)
    image = mask.astype(float)
    # Interior texture and circular particle shadows must not become the boundary.
    image[mask] -= .15*rng.random(mask.sum())
    image[(x-120)**2+(y-114)**2 < 12**2] = .1
    image += rng.normal(0, .025, image.shape)
    return 1-image if invert else image


class FindWindowTest(unittest.TestCase):
    def test_rotation_translation_noise_and_both_polarities(self):
        for angle in (0., 12., 30., 45., 60., 87.):
            for invert in (False, True):
                with self.subTest(angle=angle, invert=invert):
                    result = find_window_in_image(square_image(angle, invert=invert))
                    np.testing.assert_allclose(result.center_xy_px, [130, 122], atol=.7)
                    self.assertAlmostEqual(result.side_px, 112., delta=1.5)
                    error = abs((result.angle_deg-angle+45) % 90-45)
                    self.assertLess(error, .6)

    def test_rejects_blank_clipped_and_rectangular_regions(self):
        for image in (np.ones((256, 256)), square_image(30, center=(0, 0)),
                      np.pad(np.ones((50, 160)), ((100, 106), (48, 48)))):
            with self.assertRaises(ValueError):
                find_window_in_image(image)

    def test_selects_center_square_among_multiple_complete_windows(self):
        y, x = np.mgrid[:512, :512]
        image = np.zeros((512, 512))
        for cx, cy in ((90, 90), (270, 230), (420, 420)):
            image[(abs(x-cx) < 45) & (abs(y-cy) < 45)] = 1
        result = find_window_in_image(image)
        self.assertEqual(result.candidate_count, 3)
        np.testing.assert_allclose(result.center_xy_px, [270, 230], atol=.5)

    def test_anchor_calibration_and_roundtrip_archive(self):
        result = find_window_in_image(square_image(30))
        points = np.vstack((result.center_xy_px, result.corners_xy_px))
        stages = np.array([3e-6, -2e-6])-points*1e-7
        anchor = make_window_anchor(result, stages, [-.01, .02], [1e-4, -1e-4])
        self.assertAlmostEqual(anchor['mean_side_m'], result.side_px*1e-7)
        self.assertAlmostEqual(anchor['detector_angle_deg'], 60., delta=.5)
        for u, v, index in ((0, 0, 0), (1, 0, 1), (1, 1, 2), (0, 1, 3)):
            target = (np.array(anchor['stage_patch_origin_m'])+u*np.array(anchor['stage_patch_u_m'])+
                v*np.array(anchor['stage_patch_v_m'])+u*v*np.array(anchor['stage_patch_uv_m']))
            np.testing.assert_allclose(target, stages[index+1], atol=1e-20)
        with tempfile.TemporaryDirectory() as root:
            result.save(root, anchor=anchor)
            with np.load(Path(root)/'find_window.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['corners_xy_px'], result.corners_xy_px)
                self.assertEqual(json.loads(str(saved['metadata_json']))['anchor'], anchor)
            self.assertTrue((Path(root)/'ronchigram_window.png').is_file())

    def test_invalid_input(self):
        for image in (np.zeros(50), np.full((50, 50), np.nan), np.zeros((10, 10))):
            with self.assertRaises(ValueError):
                find_window_in_image(image)

    def test_subdivision_inset_coverage_and_serpentine_order(self):
        result = find_window_in_image(square_image(60))
        corners = result.corners_xy_px*54e-6/result.side_px
        anchor = dict(corners_stage_xy_m=corners.tolist())
        grid = subdivide_window(result, anchor)
        np.testing.assert_allclose(grid.inner_size_m, [52e-6, 52e-6], atol=1e-20)
        self.assertEqual(len(grid.cells), 25)
        u = (corners[1]-corners[0])/54e-6
        v = (corners[3]-corners[0])/54e-6
        for index, cell in enumerate(grid.cells):
            self.assertEqual(cell['id'], index+1)
            np.testing.assert_allclose(cell['size_m'], [10.4e-6]*2, atol=1e-20)
            local = (np.array(cell['corners_stage_xy_m'])-corners[0]) @ np.column_stack((u, v))
            self.assertGreaterEqual(local.min(), 1e-6-1e-18)
            self.assertLessEqual(local.max(), 53e-6+1e-18)
            if index:
                prior = grid.cells[index-1]
                self.assertEqual(abs(cell['row']-prior['row'])+abs(cell['column']-prior['column']), 1)
        with tempfile.TemporaryDirectory() as root:
            grid.save(result, root)
            self.assertTrue((Path(root)/'ronchigram_grid.png').is_file())
            with np.load(Path(root)/'window_grid.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['centers_stage_xy_m'].shape, (25, 2))
        with self.assertRaises(ValueError):
            subdivide_window(result, anchor, inset_m=30e-6)


if __name__ == '__main__':
    unittest.main()
