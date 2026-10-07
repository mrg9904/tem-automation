from __future__ import annotations

import numpy as np

class FakeMicroscope:
    def __init__(
        self,
        *,
        initial_defocus_m: float = -120e-9,
        best_defocus_m: float = 65e-9,
    ) -> None:
        self.defocus_m = initial_defocus_m
        self.best_defocus_m = best_defocus_m
        self.connected = False
        self._rng = np.random.default_rng(7)
        self.image_size_px = 128

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def get_defocus(self) -> float:
        return self.defocus_m

    def set_defocus(self, defocus_m: float) -> None:
        self.defocus_m = float(defocus_m)

    def get_fov(self) -> float:
        return 100e-9

    def acquire_haadf(self):
        size = self.image_size_px
        y, x = np.mgrid[:size, :size]
        specimen = (
            ((x > size * 0.22) & (x < size * 0.48)
             & (y > size * 0.18) & (y < size * 0.76)).astype(float)
            + 0.7 * (((x - size * 0.70) ** 2 + (y - size * 0.55) ** 2)
                     < (size * 0.14) ** 2)
        )

        defocus_error_nm = abs(self.defocus_m - self.best_defocus_m) * 1e9
        sigma_px = 0.5 + defocus_error_nm / 35.0
        frequencies_y = np.fft.fftfreq(size)[:, None]
        frequencies_x = np.fft.fftfreq(size)[None, :]
        transfer = np.exp(
            -2.0 * np.pi**2 * sigma_px**2
            * (frequencies_x**2 + frequencies_y**2)
        )
        blurred = np.fft.ifft2(np.fft.fft2(specimen) * transfer).real
        noisy = blurred + self._rng.normal(0.0, 0.005, blurred.shape)

        return noisy.astype(np.float32)
