"""Public offline ROI API. Internal modules separate detection from output."""
from .roi.config import FindROIConfig, ROI_DTYPE
from .roi.reference import ROIReferenceBank
from .roi.result import FindROIResult
from .roi.detection import find_roi, find_roi_in_image, describe_roi_candidates
from .roi.mosaic import MosaicTile, ROIMosaicResult, compose_roi_mosaic, save_roi_mosaic
# Retain private imports used by existing diagnostic tools.
from .roi.features import _features, _patch_classes, _context_scores

__all__ = ["FindROIConfig", "ROI_DTYPE", "ROIReferenceBank", "FindROIResult",
           "find_roi", "find_roi_in_image", "describe_roi_candidates", "MosaicTile",
           "ROIMosaicResult", "compose_roi_mosaic", "save_roi_mosaic"]
