import unittest
import numpy as np
from algorithms.AutoBC import auto_bc


class AutoBCTests(unittest.TestCase):
    def test_weak_foreground_is_not_percentile_clipped(self):
        data = np.zeros((100, 100))
        data[40:60, 40:60] = 100
        data[39, 40:60] = .5
        mask = data > 0
        tone = auto_bc(data, mask)
        self.assertLessEqual(tone.low, data[mask].min())
        self.assertGreaterEqual(tone.high, data[mask].max())
        np.testing.assert_array_equal(data[39, 40:60], .5)

    def test_nion_transfer_matches_png_range(self):
        data = np.arange(10000, dtype=float).reshape(100, 100)
        mask = data > 1000
        tone = auto_bc(data, mask)
        span = data.max()-data.min()
        m = tone.contrast/span
        b = 1/(2*m)-(1-tone.brightness)*span/2-data.min()
        self.assertAlmostEqual(-b, tone.low)
        self.assertAlmostEqual(1/m-b, tone.high)

    def test_constant_and_nonfinite(self):
        self.assertTrue(np.isfinite(auto_bc(np.ones((3, 3))).render(np.ones((3, 3)))).all())
        with self.assertRaises(ValueError):
            auto_bc(np.full((3, 3), np.nan))

    def test_adapter_updates_profile_only(self):
        from unittest.mock import Mock
        from adapters.nion_usim import NionUSimAdapter
        adapter = NionUSimAdapter(Mock())
        source = Mock(profile_index=2)
        adapter._hardware_source = source
        adapter._get_active_profile_frame_parameters = Mock(return_value={'pixel_time_us': 10.})
        adapter.set_display_tone(.1, 1.2)
        source.set_frame_parameters_for_profile_by_index.assert_called_once_with(
            2, {'pixel_time_us': 10., 'usim_brightness': .1, 'usim_contrast': 1.2})
