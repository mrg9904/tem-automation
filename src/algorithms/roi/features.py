from functools import lru_cache
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from adapters.cancellation import check_cancelled

def _prepare(image, fov_m, config):
    data = np.asarray(image, dtype=float)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError('Expected a finite nonempty 2-D image')
    if not np.isfinite(fov_m) or fov_m <= 0:
        raise ValueError('FoV must be finite and positive')
    if any(not np.isfinite(v) or v <= 0 for v in (config.analysis_pixel_size_nm, config.patch_size_nm, config.stride_nm)):
        raise ValueError('Physical pixel, patch and stride sizes must be positive')
    if config.stride_nm > config.patch_size_nm or config.max_reference_patches < 2:
        raise ValueError('Stride must not exceed patch size; memory size must be >= 2')
    if not np.isfinite(config.min_region_area_nm2) or config.min_region_area_nm2 < 0:
        raise ValueError('Minimum area must be finite and nonnegative')
    if not np.isfinite(config.context_radius_nm) or config.context_radius_nm <= 0 or not np.isfinite(config.broad_anomaly_factor) or config.broad_anomaly_factor < 1:
        raise ValueError('Invalid contextual scoring configuration')
    if not np.isfinite(config.min_class_threshold_fraction) or not 0<config.min_class_threshold_fraction<=1:
        raise ValueError('Invalid class threshold floor')
    if not np.isfinite(config.context_mad_multiplier) or config.context_mad_multiplier<0:
        raise ValueError('Invalid context variation tolerance')
    if any(not np.isfinite(v) or v <= 0 for v in config.normal_blur_nm):
        raise ValueError('Normal blur scales must be finite and positive')
    side = int(round(fov_m * 1e9 / config.analysis_pixel_size_nm))
    patch = max(4, int(round(config.patch_size_nm / config.analysis_pixel_size_nm)))
    if side > 2048 or min(data.shape) * side / max(data.shape) < patch:
        raise ValueError('Image FoV is too large or too small for configured physical patch scale')
    # The analysis never adds physical information unavailable in the input.
    low, high = np.percentile(data, [1, 99])
    normalized = np.clip((data-low)/max(high-low, np.finfo(float).eps), 0, 1)
    shape = tuple(max(1, int(round(n * side / max(data.shape)))) for n in data.shape)
    yy, xx = np.meshgrid(np.linspace(0, data.shape[0]-1, shape[0]),
                         np.linspace(0, data.shape[1]-1, shape[1]), indexing='ij')
    analysis = ndimage.map_coordinates(normalized, [yy, xx], order=1)
    return data.astype(np.float32), analysis, patch


@lru_cache(maxsize=16)
def _descriptor_geometry(shape):
    yy,xx = np.meshgrid(np.linspace(0,shape[0]-1,5),np.linspace(0,shape[1]-1,5),indexing='ij')
    window = np.hanning(shape[0])[:,None]*np.hanning(shape[1])[None,:]
    fy,fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(shape[0])),np.fft.fftshift(np.fft.fftfreq(shape[1])),indexing='ij')
    radius = np.hypot(fx,fy)
    masks = tuple((radius>=a)&(radius<b) for a,b in zip([0,.08,.16,.3],[.08,.16,.3,.72]))
    return yy,xx,window,masks


def _descriptor(patch):
    # Shape retains localized departures from a normal straight/corner boundary.
    smoothed = ndimage.gaussian_filter(patch, .7)
    yy,xx,window,masks = _descriptor_geometry(patch.shape)
    shape = ndimage.map_coordinates(smoothed, [yy, xx], order=1).ravel()
    gy, gx = np.gradient(smoothed)
    gradient = np.array([np.mean(gx*gx), np.mean(gy*gy), np.mean(np.abs(gx*gy))])
    # Windowed local power bands supplement morphology; not a phase classifier.
    power = np.abs(np.fft.fftshift(np.fft.fft2((patch-patch.mean())*window)))**2
    bands = [np.log1p(np.mean(power[mask])) if np.any(mask) else 0 for mask in masks]
    return np.r_[shape, patch.mean(), patch.std(), gradient, bands]


def _features(image, fov_m, config, augment=False):
    raw, analysis, side = _prepare(image, fov_m, config)
    stride = max(1, int(round(config.stride_nm / config.analysis_pixel_size_nm)))
    def positions(length):
        return np.unique(np.r_[np.arange(0, length-side+1, stride), length-side]).astype(int)
    ys, xs = positions(analysis.shape[0]), positions(analysis.shape[1])
    features = []
    for y in ys:
        check_cancelled()
        for x in xs:
            patch = analysis[y:y+side, x:x+side]
            for rotation in range(4 if augment else 1):
                features.append(_descriptor(np.rot90(patch, rotation)))
    return np.asarray(features), raw, analysis.shape, ys + (side-1)/2, xs + (side-1)/2


def _patch_classes(features):
    # Coarse context only: background, bulk, mixed/edge. No defect-shape rules.
    edge = features[:,26] >= .12
    return np.where(edge,2,np.where(features[:,25]>=.4,1,0))


def _context_scores(features, scores, ys, xs, config, thresholds):
    classes = _patch_classes(features)
    yy,xx = np.meshgrid(ys,xs,indexing='ij')
    positions = np.column_stack([yy.ravel(),xx.ravel()])*config.analysis_pixel_size_nm
    tree = cKDTree(positions)
    neighbors = tree.query_ball_point(positions,config.context_radius_nm)
    baseline = np.zeros(len(scores))
    for i, nearby in enumerate(neighbors):
        if i % 64 == 0:
            check_cancelled()
        nearby = np.asarray(nearby,dtype=int)
        distance = np.linalg.norm(positions[nearby]-positions[i],axis=1)
        similar = (np.abs(features[nearby,25]-features[i,25]) <= .08) & (np.abs(features[nearby,26]-features[i,26]) <= .04)
        same = nearby[(classes[nearby]==classes[i]) & similar & (distance>=config.patch_size_nm)]
        if len(same)<3:
            # Sparse contexts (e.g. corners) borrow the closest comparable windows
            # elsewhere in this image; candidates are never added to normal memory.
            pool = np.flatnonzero((classes==classes[i]) &
                (np.linalg.norm(positions-positions[i],axis=1)>=config.patch_size_nm))
            if len(pool)>=3:
                similarity = ((features[pool,25]-features[i,25])/.08)**2 + ((features[pool,26]-features[i,26])/.04)**2
                same = pool[np.argsort(similarity)[:min(7,len(pool))]]
        if len(same)>=3:
            median = np.median(scores[same])
            mad = np.median(np.abs(scores[same]-median))
            baseline[i] = median + config.context_mad_multiplier*1.4826*mad
    # Very strong reference evidence can survive even if an entire region differs.
    corrected = np.maximum(scores-baseline,0)
    strong = scores > config.broad_anomaly_factor*thresholds
    corrected[strong] = scores[strong]
    return corrected, baseline
