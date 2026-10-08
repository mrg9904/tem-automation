from datetime import datetime
from pathlib import Path
from uuid import uuid4
from scripts import __file__ as scripts_package_file
from adapters.nion_usim import NionUSimAdapter, nion_cancel_callback
from adapters.cancellation import cancellation_scope
from algorithms.FindROI import ROIReferenceBank, find_roi


def script_main(api_broker):
    scripts_directory=Path(scripts_package_file).resolve().parent
    path=scripts_directory/'roi_reference.npz'
    if not path.is_file():
        raise RuntimeError('Build and calibrate an approved normal reference, then save it to: '+str(path))
    reference=ROIReferenceBank.load(path)
    if reference.threshold is None:
        raise RuntimeError('Normal reference must be calibrated before ROI detection')
    output=scripts_directory/'find_roi_results'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8])
    api=api_broker.get_api(version='~1.0')
    with cancellation_scope(nion_cancel_callback(print)):
        with NionUSimAdapter(api,image_size_px=1024) as microscope:
            result=find_roi(microscope,reference,output_directory=output)
    print(f'{len(result.candidates)} ROI candidates: {output}')
