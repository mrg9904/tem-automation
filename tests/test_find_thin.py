from pathlib import Path
import tempfile
import unittest
import numpy as np
from algorithms.FindThin import (FindThinConfig, analyze_thin, find_thin, complete_particle_footprint,
                                 select_large_thin_region, save_top_thin_regions)
from adapters.base import SpectrumImageCapture


class FindThinTest(unittest.TestCase):
    def test_broad_patch_selection_ignores_a_narrow_connected_bridge(self):
        from dataclasses import replace
        cube, energy, mask, _ = self.cube()
        result = analyze_thin(cube, energy, particle_mask=mask, fov_m=200e-9)
        thin = np.zeros((40, 40), bool)
        thin[3:19, 3:19] = True
        thin[8:14, 27:35] = True
        thin[10, 18:28] = True
        selected = select_large_thin_region(replace(result, thin_mask=thin,
            t_over_lambda=np.full(thin.shape, .5), fov_m=400e-9,
            metadata={'stage_xy_m': [1e-6, 2e-6], 'image_to_stage_matrix': [[-1, 0], [0, -1]]}))
        x, y = selected['center_xy_px']
        self.assertTrue(3 < x < 19 and 3 < y < 19)
        yy, xx = np.indices(thin.shape)
        disk = (xx-x)**2+(yy-y)**2 <= selected['radius_px']**2
        self.assertTrue(thin[disk].all())
        np.testing.assert_allclose(selected['center_stage_xy_m'],
                                   np.array([1e-6, 2e-6])-selected['offset_xy_m'])

    def test_top_region_saved_using_derived_arrays_without_spectrum_cube(self):
        cube, energy, mask, _ = self.cube()
        result = analyze_thin(cube, energy, particle_mask=mask, fov_m=200e-9)
        with tempfile.TemporaryDirectory() as root:
            result.save(Path(root)/'particles/particle_0001/thickness')
            records = save_top_thin_regions(root, [dict(rank=1, particle_id=1,
                thin_area_percent=50., directory='particles/particle_0001')])
            self.assertEqual(records[0]['particle_id'], 1)
            self.assertTrue((Path(root)/records[0]['annotated_image']).exists())
            self.assertTrue((Path(root)/'top_thin_regions/regions.npz').exists())
            self.assertFalse((Path(root)/'particles/particle_0001/thickness/eels_spectrum_image.npy').exists())

    def test_nested_core_is_in_denominator_but_background_pore_is_not(self):
        labels = np.zeros((30, 30), dtype=int)
        labels[4:26, 4:26] = 1
        labels[8:22, 8:22] = 2
        labels[12:15, 12:15] = 0
        footprint = complete_particle_footprint(labels, 1)
        self.assertTrue(footprint[10, 10])
        self.assertFalse(footprint[13, 13])
        self.assertFalse(footprint[0, 0])
        energy = np.arange(-10., 101.)
        tau = np.where(labels == 1, .5, 2.)
        cube = np.zeros((30, 30, len(energy)))
        cube[..., 10] = 10000*np.exp(-tau)
        cube[..., 40] = 10000*(1-np.exp(-tau))
        result = analyze_thin(cube, energy, particle_mask=footprint, fov_m=300e-9)
        self.assertAlmostEqual(result.summary()['thin_area_percent'],
                               100*(labels == 1).sum()/footprint.sum())
        self.assertFalse(result.thin_mask[labels == 2].any())
    def cube(self):
        energy = np.arange(-10., 101.)
        cube = np.zeros((20, 20, len(energy)))
        tau = np.zeros((20, 20))
        mask = np.zeros((20, 20), bool)
        mask[3:17, 3:17] = True
        tau[3:17, 3:10] = .5
        tau[3:17, 10:17] = 1.5
        cube[..., 10] = 10000*np.exp(-tau)
        cube[..., 40] = 10000*(1-np.exp(-tau))
        return cube, energy, mask, tau

    def test_log_ratio_and_particle_area_exclude_vacuum(self):
        cube, energy, mask, tau = self.cube()
        result = analyze_thin(cube, energy, particle_mask=mask, fov_m=200e-9)
        np.testing.assert_allclose(result.t_over_lambda, tau, atol=1e-15)
        self.assertAlmostEqual(result.summary()['thin_area_percent'], 50.)
        self.assertFalse(result.thin_mask[~mask].any())
        self.assertEqual(len(result.summary()['regions']), 1)
        self.assertFalse(result.summary()['particle_touches_edge'])

    def test_missing_zlp_invalid_counts_clipping_and_archive(self):
        cube, energy, mask, tau = self.cube()
        cube[5, 5] = 0
        cube[6, 6, 10] = np.nan
        result = analyze_thin(cube, energy, particle_mask=mask, fov_m=200e-9)
        self.assertFalse(result.valid_mask[5, 5])
        self.assertFalse(result.thin_mask[6, 6])
        self.assertGreater(result.summary()['unquantified_area_percent'], 0)
        with tempfile.TemporaryDirectory() as root:
            result.save(root)
            with np.load(Path(root)/'find_thin.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['thin_mask'], result.thin_mask)
            self.assertTrue((Path(root)/'haadf_thin_annotated.png').is_file())
        with self.assertRaisesRegex(ValueError, 'zero-loss'):
            analyze_thin(cube[..., 10:], energy[10:], particle_mask=mask, fov_m=200e-9)

    def test_acquisition_contract_and_defocus(self):
        cube, energy, mask, _ = self.cube()
        class Microscope:
            def get_fov(self): return 200e-9
            def set_effective_defocus(self, value): self.defocus = value
            def acquire_eels_spectrum_image(self, **kwargs):
                self.parameters = kwargs
                return SpectrumImageCapture(cube, energy, cube.sum(axis=-1), 200e-9, {})
        microscope = Microscope()
        with tempfile.TemporaryDirectory() as root:
            result = find_thin(microscope, output_directory=root, particle_mask=mask)
        self.assertAlmostEqual(microscope.defocus, 20e-9)
        self.assertEqual(microscope.parameters['image_size_px'], 128)
        self.assertEqual(microscope.parameters['dwell_time_us'], 10.)
        self.assertEqual(microscope.parameters['dispersion_eV'], 2.)
        self.assertEqual(microscope.parameters['energy_max_eV'], 498.)
        self.assertIsNone(microscope.parameters['output_path'])
        self.assertFalse(result.metadata['spectrum_image_saved'])
        self.assertAlmostEqual(result.summary()['thin_area_percent'], 50.)

    def test_threshold_is_strict_and_footprint_clipping_is_reported(self):
        cube, energy, mask, _ = self.cube()
        cube[..., 10] = 10000/np.e
        cube[..., 40] = 10000-10000/np.e
        result = analyze_thin(cube, energy, particle_mask=np.ones((20, 20), bool), fov_m=200e-9)
        self.assertEqual(result.summary()['thin_area_percent'], 0.)
        self.assertTrue(result.summary()['particle_touches_edge'])

    def test_mapping_defocus_is_relative_to_measured_focus(self):
        cube, energy, mask, _ = self.cube()
        class Microscope:
            def get_fov(self): return 200e-9
            def set_defocus(self, value): self.defocus = value
            def set_effective_defocus(self, value): raise AssertionError('Absolute defocus must not be used')
            def acquire_eels_spectrum_image(self, **kwargs):
                return SpectrumImageCapture(cube, energy, cube.sum(axis=-1), 200e-9, {})
        microscope = Microscope()
        with tempfile.TemporaryDirectory() as root:
            result = find_thin(microscope, output_directory=root, particle_mask=mask,
                               focus_reference_m=130e-9, defocus_offset_m=200e-9)
        self.assertAlmostEqual(microscope.defocus, 330e-9)
        self.assertAlmostEqual(result.metadata['focus_reference_m'], 130e-9)
        self.assertAlmostEqual(result.metadata['defocus_offset_m'], 200e-9)
        self.assertNotIn('requested_effective_defocus_m', result.metadata)

    def test_ring_thin_region_position_is_inside_thin_area(self):
        cube, energy, mask, tau = self.cube()
        tau[mask] = .5
        tau[6:14, 6:14] = 1.5
        cube[..., 10] = 10000*np.exp(-tau)
        cube[..., 40] = 10000*(1-np.exp(-tau))
        result = analyze_thin(cube, energy, particle_mask=mask, fov_m=200e-9)
        for region in result.summary()['regions']:
            x, y = map(int, region['position_xy_px'])
            self.assertTrue(result.thin_mask[y, x])

    def test_context_footprint_counts_unmeasured_area_in_full_particle_denominator(self):
        cube, energy, _, _ = self.cube()
        cube[..., 10] = 10000*np.exp(-.5)
        cube[..., 40] = 10000*(1-np.exp(-.5))
        overview = np.zeros((40, 40)); overview[8:32, 8:32] = 5
        class Microscope:
            def get_fov(self): return 200e-9
            def set_effective_defocus(self, value): pass
            def acquire_particle_context(self, **kwargs): return overview, 400e-9
            def acquire_eels_spectrum_image(self, **kwargs):
                return SpectrumImageCapture(cube, energy, np.ones((20, 20)), 200e-9, {})
        with tempfile.TemporaryDirectory() as root:
            result = find_thin(Microscope(), output_directory=root, config=FindThinConfig(image_size_px=20))
        summary = result.summary()
        self.assertTrue(summary['has_complete_context_footprint'])
        self.assertGreater(summary['area_outside_current_fov_percent'], 0)
        self.assertAlmostEqual(summary['thin_area_percent']+summary['area_outside_current_fov_percent'], 100.)

    def test_relative_mapping_uses_focused_context_before_defocused_spectrum_image(self):
        cube, energy, _, _ = self.cube()
        overview = np.zeros((40, 40)); overview[8:32, 8:32] = 5
        events = []
        class Microscope:
            def get_fov(self): return 200e-9
            def set_defocus(self, value): self.defocus = value
            def acquire_particle_context(self, **kwargs):
                events.append(('context', self.defocus))
                return overview, 400e-9
            def acquire_eels_spectrum_image(self, **kwargs):
                events.append(('spectrum', self.defocus))
                return SpectrumImageCapture(cube, energy, np.ones((20, 20)), 200e-9, {})
        with tempfile.TemporaryDirectory() as root:
            result = find_thin(Microscope(), output_directory=root, config=FindThinConfig(image_size_px=20),
                               focus_reference_m=130e-9, defocus_offset_m=200e-9)
        self.assertEqual([event[0] for event in events], ['context', 'spectrum'])
        self.assertAlmostEqual(events[0][1], 130e-9)
        self.assertAlmostEqual(events[1][1], 330e-9)
        self.assertAlmostEqual(result.metadata['particle_context_lens_defocus_m'], 130e-9)
