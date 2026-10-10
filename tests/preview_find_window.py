"""Acquire an actual uSim Ronchigram; no model geometry enters FindWindow."""
from pathlib import Path
import numpy as np
from PIL import Image
from nion.utils import Geometry
from nion.usim_device import ThousandCathodeSample
from nionswift_plugin.usim.test.KikuchiModel_test import TestKikuchiModel


def main():
    output = Path(__file__).resolve().parents[1]/'src/scripts/find_window_results/preview'
    output.mkdir(parents=True, exist_ok=True)
    fixture = TestKikuchiModel()
    fixture.sample = ThousandCathodeSample.ThousandCathodeSample(1000)
    manager, camera, context, area = fixture.make_camera(1024)
    camera.noise.enabled = True
    manager.set_value_2d('stage_position_m', Geometry.FloatPoint(x=938e-9, y=-6820e-9))
    manager.set_value('stage_z_m', 600e-6)
    manager.set_value('C10Control', 0.)
    try:
        frame = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
        image = frame.data
        np.savez_compressed(output/'capture.npz', image=image,
            angular_offset_rad=[c.offset for c in frame.dimensional_calibrations],
            angular_scale_rad=[c.scale for c in frame.dimensional_calibrations])
        lo, hi = np.percentile(image, [1, 99])
        Image.fromarray((np.clip((image-lo)/(hi-lo), 0, 1)*255).astype('uint8')).save(output/'ronchigram.png')
        from algorithms.FindWindow import find_window_in_image, make_window_anchor
        result = find_window_in_image(image)
        targets = []
        for x, y in np.vstack((result.center_xy_px, result.corners_xy_px)):
            delta = camera.stage_displacement_for_pixel(Geometry.FloatPoint(x=x, y=y), image.shape)
            targets.append(np.array([938e-9, -6820e-9])-[delta.x, delta.y])
        anchor = make_window_anchor(result, targets,
            [c.offset for c in frame.dimensional_calibrations], [c.scale for c in frame.dimensional_calibrations])
        result.save(output, anchor=anchor)
        print(result.summary())
        print(anchor)
        print(output)
    finally:
        camera.close()


if __name__ == '__main__':
    main()
