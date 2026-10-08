from dataclasses import dataclass
import numpy as np

ROI_DTYPE = np.dtype([
    ('id', '<i4'), ('center_x_px', '<f8'), ('center_y_px', '<f8'),
    ('offset_x_m', '<f8'), ('offset_y_m', '<f8'),
    ('x0_px', '<i4'), ('y0_px', '<i4'), ('x1_px', '<i4'), ('y1_px', '<i4'),
    ('area_nm2', '<f8'), ('score_max', '<f8'), ('score_mean', '<f8'), ('touches_frame', '?'),
])


@dataclass(frozen=True)
class FindROIConfig:
    analysis_pixel_size_nm: float = 0.2
    patch_size_nm: float = 5.0
    stride_nm: float = 2.0
    max_reference_patches: int = 2048
    min_region_area_nm2: float = 1.0
    normal_blur_nm: tuple[float, ...] = (0.3, 0.6)
    context_radius_nm: float = 15.0
    broad_anomaly_factor: float = 20.0
    min_class_threshold_fraction: float = 0.5
    context_mad_multiplier: float = 2.5
