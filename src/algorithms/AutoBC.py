"""Automatic HAADF display scaling; quantitative data and masks stay unchanged."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class AutoBCResult:
    low: float
    high: float
    brightness: float
    contrast: float

    def render(self, image):
        values = np.nan_to_num(np.asarray(image, dtype=float), nan=self.low,
                               posinf=self.high, neginf=self.low)
        return np.rint(255*np.clip((values-self.low)/(self.high-self.low), 0, 1)).astype(np.uint8)

    def metadata(self):
        return asdict(self)


def auto_bc(image, particle_mask=None):
    """Keep every finite foreground value within the display range.

    A supplied footprint protects weak boundary pixels from percentile clipping.
    Without a mask, the full finite range is retained. This cannot recover signal
    below noise or a particle outside the acquired field of view.
    """
    data = np.asarray(image, dtype=float)
    if data.ndim != 2 or not np.isfinite(data).any():
        raise ValueError('AutoBC requires a finite two-dimensional image')
    finite = np.isfinite(data)
    low, high = float(data[finite].min()), float(data[finite].max())
    original_low, original_high = low, high
    if particle_mask is not None:
        mask = np.asarray(particle_mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError('Particle mask must match image shape')
        background = data[finite & ~mask]
        foreground = data[finite & mask]
        if background.size and foreground.size:
            # Suppress rare negative background excursions, preserving ALL
            # foreground intensities rather than clipping a fixed percentile.
            low = min(float(np.percentile(background, 1)), float(foreground.min()))
    if high <= low:
        high = low+max(1., abs(low)*1e-6)
    contrast = (original_high-original_low)/(high-low) if original_high > original_low else 1.
    brightness = ((original_low+original_high-low-high)/(original_high-original_low)
                  if original_high > original_low else 0.)
    return AutoBCResult(low, high, brightness, contrast)
