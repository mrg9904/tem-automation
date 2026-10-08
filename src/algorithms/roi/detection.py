import numpy as np
from scipy import ndimage
from .config import ROI_DTYPE
from .features import _features, _patch_classes, _context_scores
from .reference import ROIReferenceBank
from .result import FindROIResult

def find_roi_in_image(image, *, fov_m, reference: ROIReferenceBank, threshold=None, output_directory=None):
    """Find candidates; scores are standardized feature distances, not probabilities."""
    explicit_threshold = threshold is not None
    threshold = reference.threshold if threshold is None else threshold
    if threshold is None or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Calibrate the normal reference or provide an explicit positive threshold')
    features, raw, shape, ys,xs = _features(image,fov_m,reference.config)
    raw_scores, nearest = reference.query_features(features)
    patch_thresholds = (np.full(len(features), threshold) if explicit_threshold or reference.class_thresholds is None
                        else reference.class_thresholds[_patch_classes(features)])
    scores, context = _context_scores(features, raw_scores, ys, xs, reference.config, patch_thresholds)
    scores = scores.reshape(len(ys),len(xs))
    yy,xx = np.meshgrid(np.interp(np.linspace(0,shape[0]-1,raw.shape[0]),ys,np.arange(len(ys))),
                        np.interp(np.linspace(0,shape[1]-1,raw.shape[1]),xs,np.arange(len(xs))),indexing='ij')
    score_map = ndimage.map_coordinates(scores,[yy,xx],order=1).astype(np.float32)
    def expand(values):
        return ndimage.map_coordinates(values.reshape(len(ys),len(xs)),[yy,xx],order=1).astype(np.float32)
    threshold_map = expand(patch_thresholds)
    reference_map = expand(raw_scores)
    context_map = expand(context)
    labels,count = ndimage.label(score_map>threshold_map,structure=np.ones((3,3)))
    pixel_nm = fov_m*1e9/max(raw.shape)
    candidates=[]
    output_labels = np.zeros(raw.shape,dtype=np.int32)
    for identifier, region in enumerate(ndimage.find_objects(labels), start=1):
        mask=labels[region]==identifier
        area=int(mask.sum())
        if area*pixel_nm**2 < reference.config.min_region_area_nm2:
            continue
        y,x=np.nonzero(mask);y=y+region[0].start;x=x+region[1].start
        cx,cy=float(x.mean()),float(y.mean())
        identifier_out=len(candidates)+1
        output_labels[y,x]=identifier_out
        values=score_map[y,x]
        candidates.append((identifier_out,cx,cy,(cx+.5-raw.shape[1]/2)*pixel_nm*1e-9,
            (cy+.5-raw.shape[0]/2)*pixel_nm*1e-9,region[1].start,region[0].start,
            region[1].stop,region[0].stop,area*pixel_nm**2,float(values.max()),float(values.mean()),
            bool((x==0).any() or (y==0).any() or (x==raw.shape[1]-1).any() or (y==raw.shape[0]-1).any())))
    result=FindROIResult(raw,float(fov_m),score_map,output_labels,np.asarray(candidates,dtype=ROI_DTYPE),float(threshold),reference.model_id, reference_score_map=reference_map, context_baseline_map=context_map, threshold_map=threshold_map)
    result = describe_roi_candidates(result, reference,
        feature_batch=(features, raw, shape, ys, xs), reference_matches=(raw_scores, nearest))
    if output_directory is not None:
        result.save(output_directory)
    return result


def find_roi(microscope, reference, *, output_directory=None, threshold=None):
    fov=microscope.get_fov()
    return find_roi_in_image(microscope.acquire_haadf(),fov_m=fov,reference=reference,
                             threshold=threshold,output_directory=output_directory)


def describe_roi_candidates(result, reference, *, feature_batch=None, reference_matches=None):
    """Describe measured feature departures, without assigning defect identities."""
    from dataclasses import replace
    if not len(result.candidates):
        return replace(result, descriptions=())
    features, _, shape, ys, xs = (_features(result.image, result.fov_m, reference.config)
        if feature_batch is None else feature_batch)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    px = xx.ravel() * (result.image.shape[1]-1) / max(shape[1]-1, 1)
    py = yy.ravel() * (result.image.shape[0]-1) / max(shape[0]-1, 1)
    distances, nearest = (reference.query_features(features)
        if reference_matches is None else reference_matches)
    descriptions = []
    for roi in result.candidates:
        within = (px>=roi['x0_px']) & (px<roi['x1_px']) & (py>=roi['y0_px']) & (py<roi['y1_px'])
        indices = np.flatnonzero(within)
        index = (indices[np.argmax(distances[indices])] if len(indices) else
                 np.argmin((px-roi['center_x_px'])**2 + (py-roi['center_y_px'])**2))
        delta = ((features[index]-reference.features[nearest[index]])/reference.scale*reference.weights)**2
        groups = [delta[:25].sum()+delta[27:30].sum(), delta[26]+delta[30:].sum(), delta[25]]
        words = ['shape variation', 'texture variation', 'brightness variation']
        chosen = [words[i] for i in np.argsort(groups)[::-1] if groups[i] >= max(groups)*.35][:2]
        prefix = 'weak anomaly; ' if roi['score_max'] < 2*result.threshold else ''
        descriptions.append(prefix + ', '.join(chosen))
    return replace(result, descriptions=tuple(descriptions))
