"""Nion Swift entry: initialize the grid view and save its central-window anchor."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
from scripts import __file__ as scripts_package_file
from adapters.nion_usim_ronchigram import NionUSimRonchigramAdapter
from adapters.nion_usim import nion_cancel_callback
from adapters.cancellation import cancellation_scope, WorkflowCancelled
from algorithms.FindWindow import find_window_in_image, make_window_anchor


def script_main(api_broker):
    directory = Path(scripts_package_file).resolve().parent/'find_window_results'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with cancellation_scope(nion_cancel_callback(print)):
            with NionUSimRonchigramAdapter(api_broker.get_api(version='~1.0'), timeout_s=120.) as microscope:
                microscope.initialize_window_view()
                image = microscope.acquire_ronchigram()
                np.savez_compressed(directory/'capture.npz', image=image,
                    stage_xy_m=microscope.get_stage_position(), stage_z_m=600e-6)
                result = find_window_in_image(image)
                points = np.vstack((result.center_xy_px, result.corners_xy_px))
                anchor = make_window_anchor(result, *microscope.window_coordinates(points))
                result.save(directory, anchor=anchor)
        print(f"Window found: side {anchor['mean_side_m']*1e6:.2f} um, detector angle {anchor['detector_angle_deg']:.2f} deg (mod 90).")
    except WorkflowCancelled:
        print('FindWindow cancelled.')
    finally:
        print(f'Images and data: {directory}')
