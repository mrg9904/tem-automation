"""uSim spectrum-image acquisition; device-specific physics stays in uSim."""
import numpy as np
from adapters.base import SpectrumImageCapture
from adapters.nion_usim import NionUSimAdapter
from adapters.cancellation import check_cancelled


class NionUSimEELSAdapter(NionUSimAdapter):
    def set_effective_defocus(self, defocus_m):
        if not np.isfinite(defocus_m):
            raise ValueError('Defocus must be finite')
        self.set_defocus(float(self._require_instrument().get_control_output('stage_z_m'))+defocus_m)

    def acquire_eels_spectrum_image(self, *, image_size_px, dwell_time_us, output_path,
                                    dispersion_eV=2., energy_min_eV=-10., energy_max_eV=498.):
        from nion.utils import Registry
        instrument = Registry.get_component('stem_controller')
        if instrument is None or instrument.instrument_id != self._instrument_id:
            raise RuntimeError('Connected uSim instrument is unavailable')
        simulator = instrument.value_manager.eels_camera.simulator
        if not callable(getattr(simulator, 'acquire_spectrum_image', None)):
            raise RuntimeError('Update uSim to enable its batch spectrum-imaging interface')
        check_cancelled()
        parameters = self._get_active_profile_frame_parameters()
        parameters.update(pixel_size=(image_size_px, image_size_px), pixel_time_us=dwell_time_us)
        if not np.isfinite([dispersion_eV, energy_min_eV, energy_max_eV]).all() or dispersion_eV <= 0 or energy_min_eV >= energy_max_eV:
            raise ValueError('Invalid spectral window')
        channels = int(np.floor((energy_max_eV-energy_min_eV)/dispersion_eV))+1
        if not 2 <= channels <= instrument.camera_sensor_dimensions('eels')[1]:
            raise ValueError('Requested energy channel count exceeds the uSim detector')
        manager = instrument.value_manager
        original = manager.energy_offset_eV, manager.energy_per_channel_eV
        try:
            manager.energy_offset_eV = energy_min_eV
            manager.energy_per_channel_eV = dispersion_eV
            spectra, energies, metadata = simulator.acquire_spectrum_image(
                parameters, output_path=output_path, energy_channel_count=channels, check_cancelled=check_cancelled)
            with self.acquisition_settings(image_size_px=image_size_px, dwell_time_us=dwell_time_us):
                haadf = self.acquire_haadf()
        finally:
            manager.energy_offset_eV, manager.energy_per_channel_eV = original
        metadata.update(stage_xy_m=self.get_stage_position(), stage_z_m=instrument.GetVal('stage_z_m'),
                        image_to_stage_matrix=(-self.get_coordinate_transform().rotation).tolist(),
                        scan_rotation_rad=self.get_scan_rotation(), haadf_acquisition='same view, separate simulated scan')
        metadata['spectral_window_truncated'] = bool(energy_max_eV < 4000.)
        return SpectrumImageCapture(spectra, energies, haadf, self.get_fov(), metadata)
