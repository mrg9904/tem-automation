from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from adapters.cancellation import check_cancelled
from .config import FindROIConfig
from .features import _features, _patch_classes

class ROIReferenceBank:
    """Explicit normal feature memory plus a held-out-normal calibrated threshold."""
    def __init__(self, config=FindROIConfig()):
        self.config = config
        self.features = None
        self.center = None
        self.scale = None
        self.threshold = None
        self.class_thresholds = None
        self.weights = np.ones(34)
        self.model_id = uuid4().hex
        self.normal_image_count = 0
        self.calibration_image_count = 0

    def fit(self, normal_samples):
        """normal_samples is a sequence of (image, fov_m) approved as normal."""
        self.features = None
        self.normal_image_count = 0
        return self.update(normal_samples)

    def update(self, normal_samples):
        samples = list(normal_samples)
        if not samples:
            raise ValueError('Provide approved normal images')
        blocks = []
        nuisance = []
        for image, fov in samples:
            original = _features(image, fov, self.config, augment=True)[0]
            blocks.append(original)
            for blur_nm in self.config.normal_blur_nm:
                check_cancelled()
                blurred = ndimage.gaussian_filter(np.asarray(image, dtype=float),
                    blur_nm / (fov*1e9/max(np.asarray(image).shape)))
                variants = _features(blurred, fov, self.config, augment=True)[0]
                nuisance.append((variants-original)/np.maximum(np.std(original,axis=0),.05))
                blocks.append(variants)
        if self.features is not None:
            blocks.insert(0, self.features)
        memory = np.concatenate(blocks)
        self.center = np.mean(memory, axis=0)
        self.scale = np.maximum(np.std(memory, axis=0), .05)
        # Downweight sensitivity to allowed normal blur; never learn from test candidates.
        if nuisance:
            variation = np.sqrt(np.mean(np.concatenate(nuisance)**2,axis=0))
            self.weights = np.clip(1/np.sqrt(1+variation**2), .05, 1)
        else:
            self.weights = np.ones(34)
        # Bound memory with diverse quantized descriptors; no claim of PatchCore coreset equivalence.
        _, indices = np.unique(np.round((memory-self.center)/self.scale, 2), axis=0, return_index=True)
        # Keep feature-space ordering so rare shapes are not discarded by image order.
        if len(indices) > self.config.max_reference_patches:
            pool = indices[np.linspace(0,len(indices)-1,min(len(indices),8192)).astype(int)]
            vectors = ((memory[pool]-self.center)/self.scale*self.weights).astype(np.float32)
            distance = np.full(len(pool),np.inf)
            selected = []
            current = int(np.argmax(np.sum(vectors**2,axis=1)))
            # Reserve diverse anchors before filling the remaining memory uniformly.
            for _ in range(min(256,self.config.max_reference_patches//2)):
                selected.append(pool[current])
                distance = np.minimum(distance,np.sum((vectors-vectors[current])**2,axis=1))
                distance[current] = -1
                current = int(np.argmax(distance))
            remaining = np.setdiff1d(indices,np.asarray(selected),assume_unique=True)
            count = self.config.max_reference_patches-len(selected)
            fill = remaining[np.linspace(0,len(remaining)-1,count).astype(int)]
            indices = np.r_[selected,fill]
        self.features = memory[indices]
        self.normal_image_count += len(samples)
        self.class_thresholds = None
        self.threshold = None  # Changing normal memory invalidates previous calibration.
        self.calibration_image_count = 0
        self.model_id = uuid4().hex
        return self

    def query_features(self, features):
        if self.features is None or not len(self.features):
            raise ValueError('Normal reference bank has not been fitted')
        signature = tuple(hash(array.tobytes()) for array in
            (self.features,self.center,self.scale,self.weights))
        if getattr(self, '_index_signature', None) != signature:
            self._index = cKDTree((self.features-self.center)/self.scale*self.weights)
            self._index_signature = signature
        distances, nearest = self._index.query((features-self.center)/self.scale*self.weights)
        return distances / np.sqrt(features.shape[1]), nearest

    def _score(self, features):
        return self.query_features(features)[0]

    def calibrate(self, validation_normal_samples, *, quantile=.995, margin=1.1):
        """Use independent approved normal images, including normal corners.

        Quantile is a patch-score calibration statistic, not guaranteed image
        false-positive rate. Training-image calibration is optimistically biased.
        """
        if not 0 < quantile <= 1 or not np.isfinite(margin) or margin < 1:
            raise ValueError('Invalid calibration quantile or margin')
        samples = list(validation_normal_samples)
        if not samples:
            raise ValueError('Provide independent normal validation images')
        batches = []
        for im, fov in samples:
            batches.append(_features(im,fov,self.config))
            # Held-out normals also calibrate the declared normal focus envelope,
            # including scales between the training blur examples.
            scales = sorted(set(v for blur in self.config.normal_blur_nm for v in (blur*.5,blur)))
            for blur_nm in scales:
                check_cancelled()
                variant = ndimage.gaussian_filter(np.asarray(im,dtype=float),
                    blur_nm/(fov*1e9/max(np.asarray(im).shape)))
                batches.append(_features(variant,fov,self.config))
        features = np.concatenate([batch[0] for batch in batches])
        classes = _patch_classes(features)
        scores = self._score(features)
        fallback = max(float(np.quantile(scores,quantile))*margin, 1e-6)
        self.class_thresholds = np.array([max(float(np.quantile(scores[classes==kind],quantile))*margin,1e-6)
            if np.any(classes==kind) else fallback for kind in range(3)])
        self.threshold = float(self.class_thresholds.max())
        self.class_thresholds = np.maximum(self.class_thresholds, self.threshold*self.config.min_class_threshold_fraction)
        self.calibration_image_count = len(samples)
        return self.threshold

    def save(self, path):
        if self.features is None:
            raise ValueError('Cannot save an unfitted reference')
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, schema_version=2, method='local_spatial_fft_context_v2',
                features=self.features, center=self.center, scale=self.scale, weights=self.weights,
                class_thresholds=np.full(3,np.nan) if self.class_thresholds is None else self.class_thresholds,
                threshold=np.nan if self.threshold is None else self.threshold,
                config_json=json.dumps(asdict(self.config)), model_id=self.model_id,
                normal_image_count=self.normal_image_count, calibration_image_count=self.calibration_image_count)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if str(d['method']) != 'local_spatial_fft_context_v2' or int(d['schema_version']) != 2:
                raise ValueError('ROI reference needs rebuilding and calibration for contextual scoring v2')
            result = cls(FindROIConfig(**json.loads(str(d['config_json']))))
            result.features, result.center, result.scale = d['features'].copy(),d['center'].copy(),d['scale'].copy()
            if result.features.ndim != 2 or result.features.shape[1] != 34 or result.center.shape != (34,) or result.scale.shape != (34,) or not np.isfinite(result.center).all() or not np.isfinite(result.features).all() or not np.isfinite(result.scale).all() or (result.scale<=0).any():
                raise ValueError('Invalid reference feature arrays')
            result.weights = d['weights'].copy()
            thresholds = d['class_thresholds'].copy()
            if result.weights.shape != (34,) or not np.isfinite(result.weights).all() or (result.weights<=0).any() or thresholds.shape != (3,):
                raise ValueError('Invalid scoring parameters')
            result.class_thresholds = thresholds if np.isfinite(thresholds).all() and (thresholds>0).all() else None
            threshold = float(d['threshold'])
            if np.isfinite(threshold) and (threshold<=0 or result.class_thresholds is None):
                raise ValueError('Invalid calibrated thresholds')
            result.threshold = threshold if np.isfinite(threshold) else None
            result.model_id = str(d['model_id'])
            result.normal_image_count = int(d['normal_image_count'])
            result.calibration_image_count = int(d['calibration_image_count'])
        return result
