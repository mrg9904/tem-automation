"""uSim camera acquisition and native image-to-stage mapping."""
import numpy as np
from adapters.nion_usim import NionUSimAdapter
from adapters.cancellation import check_cancelled


class NionUSimRonchigramAdapter(NionUSimAdapter):
    def connect(self):
        super().connect()
        self._camera_source = self._api.get_hardware_source_by_id('usim_ronchigram_camera', version='~1.0')
        if self._camera_source is None:
            raise RuntimeError('uSim Ronchigram camera is unavailable')

    def close(self):
        self._camera_source = None
        super().close()

    def _native_instrument(self):
        from nion.utils import Registry
        instrument = Registry.get_component('stem_controller')
        if instrument is None or instrument.instrument_id != self._instrument_id:
            raise RuntimeError('Registered uSim instrument does not match the connected adapter')
        return instrument

    def initialize_window_view(self, *, sample='1000CathodeParticleOnCarbon', x_m=938e-9, y_m=-6820e-9, z_m=600e-6):
        check_cancelled()
        if not np.isfinite([x_m, y_m, z_m]).all():
            raise ValueError('Stage coordinates must be finite')
        from nion.utils import Geometry
        instrument = self._native_instrument()
        generator = instrument.scan_data_generator
        choices = [(index, generator.sample_titles[index]) for index in generator.selectable_sample_indices]
        index = next((index for index, title in choices if title == sample), None)
        if index is None:
            raise ValueError(f'uSim sample is unavailable: {sample}')
        generator.sample_index = index
        self.set_stage_position(x_m, y_m)
        self._require_instrument().set_control_output('stage_z_m', z_m)
        self.set_defocus(0.)
        instrument.probe_position = Geometry.FloatPoint(.5, .5)

    def acquire_ronchigram(self):
        """Record a full detector frame; scan FoV never substitutes for calibration."""
        source = self._camera_source
        if source is None:
            raise RuntimeError('Adapter is not connected')
        self._wait_until_recording_finishes(source)
        parameters = dict(source.get_frame_parameters())
        parameters.update(binning=1, exposure_ms=100.)
        parameters.pop('readout_area', None)
        frames = self._record_cancellable(source, parameters)
        self._wait_until_recording_finishes(source)
        if not frames or frames[0] is None:
            raise RuntimeError('No Ronchigram frame returned')
        frame = frames[0]
        self._capture_shape = frame.data.shape
        self._capture_stage = self.get_stage_position()
        self._capture_controls = self._view_controls()
        self._capture_calibrations = frame.dimensional_calibrations
        return np.asarray(frame.data, dtype=np.float32).copy()

    def _view_controls(self):
        instrument = self._require_instrument()
        names = ('C10', 'stage_z_m', 'stage_tilt_rad.x', 'stage_tilt_rad.y', 'beam_shift_m.x', 'beam_shift_m.y')
        return tuple(instrument.get_control_output(name) for name in names)

    def window_coordinates(self, points_xy_px):
        """Map fitted center/corners with the same ray model as mouse navigation.

        Returns stage targets, not specimen positions. Requires unchanged view
        and a valid native camera cache; never silently guesses a magnification.
        """
        from nion.utils import Geometry
        if self.get_stage_position() != self._capture_stage or self._view_controls() != self._capture_controls:
            raise RuntimeError('View changed after Ronchigram acquisition; capture again')
        simulator = self._native_instrument().value_manager.ronchigram_camera.simulator
        targets = []
        for x, y in points_xy_px:
            check_cancelled()
            delta = simulator.stage_displacement_for_pixel(Geometry.FloatPoint(x=float(x), y=float(y)), self._capture_shape)
            if delta is None:
                raise RuntimeError('Ronchigram ray mapping is unavailable or stale; capture again')
            targets.append(np.asarray(self._capture_stage)-[delta.x, delta.y])
        calibration = self._capture_calibrations
        if len(calibration) != 2 or any(c.units != 'rad' for c in calibration):
            raise RuntimeError('Ronchigram detector requires angular calibration in radians')
        return np.asarray(targets), np.asarray([c.offset for c in calibration]), np.asarray([c.scale for c in calibration])
