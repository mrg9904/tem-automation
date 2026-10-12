"""Metric, inverse tilt, image-detection and information-boundary checks."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from scipy.spatial.transform import Rotation
from algorithms.crystallography import load_crystal_geometry, stage_rotation, electron_wavelength_angstrom
from algorithms.FindZone import (FindZoneConfig, detect_bands, preprocess, zone_suggestions, index_bands,
                                 PatternRenderer, equivalent_to_any, nearest_small_index_zones)


CIF = Path(__file__).resolve().parents[2]/'nionswift-usim/nion/usim_device/samples/NNMTO_pristine.cif'


@unittest.skipUnless(CIF.exists(), 'Local CIF fixture is unavailable')
class FindZoneTests(unittest.TestCase):
    def setUp(self):
        self.crystal = load_crystal_geometry(str(CIF))

    def test_nonorthogonal_metric_and_systematic_absences(self):
        a, b = self.crystal.cell[:, 0], self.crystal.cell[:, 1]
        self.assertAlmostEqual(np.rad2deg(np.arccos(a@b/np.linalg.norm(a)/np.linalg.norm(b))), 120.)
        self.assertNotIn([0, 0, 1], self.crystal.hkl.tolist())
        self.assertIn([0, 0, 3], self.crystal.hkl.tolist())
        index = self.crystal.hkl.tolist().index([0, 0, 3])
        self.assertAlmostEqual(self.crystal.d_angstrom[index], 16.033/3)
        for symmetry in self.crystal.symmetries:
            np.testing.assert_allclose(symmetry.T@symmetry, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(symmetry), 1.)

    def test_inverse_tilts_align_a_valid_equivalent_axis_from_nonzero_tilt(self):
        current = np.deg2rad([7., -12.])
        body = Rotation.from_euler('xyz', [27., 43., 61.], degrees=True).as_matrix()
        matrix = stage_rotation(*current)@body
        rows = zone_suggestions(matrix, self.crystal, current)
        self.assertTrue(rows)
        for row in rows:
            target = np.deg2rad(row['target_tilt_deg'])
            np.testing.assert_allclose(target-current, np.deg2rad([row['delta_tx_deg'], row['delta_ty_deg']]))
            valid = False
            for label in row['possible_zone_axes']:
                axis = np.array([int(value) for value in label[1:-1]])
                direction = self.crystal.cell@axis
                direction /= np.linalg.norm(direction)
                for symmetry in self.crystal.symmetries:
                    aligned = stage_rotation(*target)@body@symmetry@direction
                    valid |= abs(aligned[2]) > np.cos(np.deg2rad(.1))
            self.assertTrue(valid, row)

    def test_negative_calibration_scales_are_applied_once(self):
        image = np.arange(128*128, dtype=float).reshape(128, 128)+1
        calibration = dict(offset_yx_rad=[.1, .1], scale_yx_rad=[-.2/127, -.2/127])
        canonical, _, _, _, axes = preprocess(image, calibration, FindZoneConfig(analysis_size=128))
        self.assertGreater(canonical[0, 0], canonical[-1, -1])
        self.assertTrue(np.diff(axes[0]).min() > 0)
        self.assertTrue(np.diff(axes[1]).min() > 0)

    def test_single_band_family_cannot_produce_an_orientation(self):
        bands = [dict(normal=[1., 0., 0.]), dict(normal=[1., .01, 0.]), dict(normal=[1., -.01, 0.])]
        candidates, hypotheses = index_bands(bands, np.ones((64, 64)), np.ones((64, 64), bool),
            (np.linspace(-.1, .1, 64),)*2, self.crystal, electron_wavelength_angstrom(100000.),
            FindZoneConfig(), lambda message: None)
        self.assertEqual(candidates, [])
        self.assertEqual(hypotheses, [])

    def test_instrument_capture_parameters_do_not_access_orientation_truth(self):
        from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
        class PhaseOnly:
            crystal_cif_path = str(CIF)
            def __getattr__(self, name):
                raise AssertionError('Forbidden orientation-truth access: '+name)
        native = SimpleNamespace(scan_data_generator=SimpleNamespace(sample=PhaseOnly()),
            GetVal=lambda name: 100000. if name == 'EHT' else 0.,
            GetVal2D=lambda name: SimpleNamespace(x=0., y=0.))
        adapter = NionUSimRonchigramAdapter(Mock())
        adapter._native_instrument = lambda: native
        adapter.get_stage_position = lambda: (0., 0.)
        adapter.get_defocus = lambda: 100e-9
        adapter._capture_kikuchi_enabled = True
        adapter._capture_calibrations = [SimpleNamespace(units='rad', offset=.1, scale=-.001)]*2
        parameters = adapter.zone_capture_parameters()
        self.assertEqual(parameters['cif_path'], str(CIF))
        self.assertNotIn('crystal_rotation_matrix', parameters)

    def test_no_contrast_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'contrast'):
            preprocess(np.ones((128, 128)), dict(offset_yx_rad=[-.1, -.1], scale_yx_rad=[.002, .002]), FindZoneConfig())

    def test_vectorized_equivalence_matches_rotation_distance(self):
        original = Rotation.from_euler('xyz', [27., 43., 61.], degrees=True).as_matrix()
        for angle in (0., 1., 3., 9.):
            candidate = Rotation.from_euler('x', angle, degrees=True).as_matrix() @ original @ self.crystal.symmetries[-1]
            reference = min(Rotation.from_matrix(candidate @ symmetry @ original.T).magnitude()
                            for symmetry in self.crystal.symmetries) < np.deg2rad(2.)
            self.assertEqual(equivalent_to_any(candidate, [original], self.crystal.symmetries, np.deg2rad(2.)), reference)

    def test_cached_chunk_renderer_preserves_scalar_pattern(self):
        axes = (np.linspace(-.1, .1, 64), np.linspace(-.1, .1, 72))
        matrix = Rotation.from_euler('xyz', [27., 43., 61.], degrees=True).as_matrix()
        wavelength = electron_wavelength_angstrom(100000.)
        renderer = PatternRenderer(self.crystal, axes, wavelength)
        yy, xx = np.meshgrid(*axes, indexing='ij')
        rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        normals = self.crystal.normals @ matrix.T
        half = wavelength/(2*self.crystal.d_angstrom)
        for sigma in (.0007, .003):
            expected = np.zeros(xx.shape)
            for index in np.flatnonzero(abs(normals[:, 2]) < np.hypot(.1, .1)+half+4*sigma):
                distance = rays @ normals[index]
                for sign in (-1, 1):
                    expected += sign*self.crystal.weights[index]*np.exp(-.5*((distance-sign*half[index])/sigma)**2)
            np.testing.assert_allclose(renderer(matrix, sigma), expected, atol=1e-12)

    def test_signed_index_search_identifies_210_and_its_family(self):
        z = self.crystal.cell @ [2, 1, 0]
        z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
        matrix = np.stack([x, np.cross(z, x), z])
        nearest = nearest_small_index_zones(matrix, self.crystal)[0]
        self.assertEqual(nearest['zone_axis'], [2, 1, 0])
        self.assertLess(nearest['beam_axis_angle_deg'], 1e-5)
        self.assertIn([1, -1, 0], nearest['equivalent_zone_axes'])

    def test_analytic_pattern_jacobian_matches_finite_differences(self):
        from algorithms.FindZone import _standardize, _standardize_with_jacobian
        renderer = PatternRenderer(self.crystal, (np.linspace(-.1, .1, 48),)*2,
                                   electron_wavelength_angstrom(100000.))
        base = Rotation.from_euler('xyz', [27., 43., 61.], degrees=True).as_matrix()
        parameters = np.array([.012, -.008, .003, np.log(.0015)])
        matrix = Rotation.from_rotvec(parameters[:3]).as_matrix()@base
        image, derivative = renderer.value_and_jacobian(matrix, np.exp(parameters[3]), parameters[:3])
        mask = np.ones(image.shape, bool); mask[:3] = False
        _, derivative = _standardize_with_jacobian(image, derivative, mask)
        for axis in range(4):
            a, b = parameters.copy(), parameters.copy()
            a[axis] += 1e-6; b[axis] -= 1e-6
            finite = (_standardize(renderer(Rotation.from_rotvec(a[:3]).as_matrix()@base, np.exp(a[3])), mask)-
                      _standardize(renderer(Rotation.from_rotvec(b[:3]).as_matrix()@base, np.exp(b[3])), mask))/(2e-6)
            self.assertLess(np.linalg.norm(derivative[..., axis]-finite)/np.linalg.norm(finite), 1e-5)
