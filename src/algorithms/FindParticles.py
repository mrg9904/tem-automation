"""Find separated particles in a HAADF field of view and save NumPy arrays.

Coordinates use pixel centers (x=column, y=row). Offsets are image-axis
coordinates relative to the FoV center, not absolute stage coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import json
from adapters.cancellation import check_cancelled
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

from adapters.base import Microscope


PARTICLE_DTYPE = np.dtype([
    ("id", "<i4"),
    ("center_x_px", "<f8"),
    ("center_y_px", "<f8"),
    ("offset_x_m", "<f8"),
    ("offset_y_m", "<f8"),
    ("area_px", "<i8"),
    ("circle_diameter_px", "<f8"),
    ("circle_diameter_m", "<f8"),
    ("touches_edge", "?"),
])


@dataclass(frozen=True)
class FindParticlesConfig:
    """Explicit thresholds override local SNR; 'otsu' retains legacy behavior."""

    gaussian_sigma_px: float = 1.0
    min_area_px: int = 16
    threshold: float | None = None
    bright_particles: bool = True
    exclude_edge_particles: bool = False
    threshold_method: str = "local_snr"
    background_sigma_m: float = 160e-9
    background_clip_percentile: float = 85.
    seed_snr: float = 5.
    grow_snr: float = 3.
    recover_edge_centers: bool = True
    maximum_context_expansions: int = 2


@dataclass(frozen=True)
class FindParticlesResult:
    particles: np.ndarray
    labels: npt.NDArray[np.int32]
    image: npt.NDArray[np.float32]
    fov_m: float
    threshold: float
    config: FindParticlesConfig
    background: np.ndarray | None = None
    noise_scale: np.ndarray | None = None
    segmentation_method: str = "otsu"
    center_status: np.ndarray | None = None
    visible_centers_xy_px: np.ndarray | None = None
    full_segmented_area_m2: np.ndarray | None = None
    context_metadata: tuple = ()

    @property
    def center_statuses(self):
        return (np.where(self.particles['touches_edge'], 'clipped', 'complete')
                if self.center_status is None else self.center_status)

    def save(self, path: str | Path) -> Path:
        """Save a versioned archive that loads with allow_pickle=False.

        An existing file at this path is replaced. The parent must exist.
        """
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Particle array file must have a .npz extension")
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.int32(2),
                particles=self.particles,
                labels=self.labels,
                image=self.image,
                image_shape_px=np.asarray(self.image.shape, dtype=np.int64),
                fov_m=np.float64(self.fov_m),
                pixel_size_m=np.float64(self.fov_m / max(self.image.shape)),
                threshold=np.float64(self.threshold),
                threshold_role=np.asarray('bright-core seed threshold; local_snr uses background/noise maps' if self.segmentation_method == 'local_snr' else 'global segmentation threshold'),
                gaussian_sigma_px=np.float64(self.config.gaussian_sigma_px),
                min_area_px=np.int64(self.config.min_area_px),
                bright_particles=np.bool_(self.config.bright_particles),
                exclude_edge_particles=np.bool_(self.config.exclude_edge_particles),
                threshold_method=np.asarray(self.segmentation_method),
                background=np.asarray(self.background if self.background is not None else np.empty((0, 0))),
                noise_scale=np.asarray(self.noise_scale if self.noise_scale is not None else np.empty((0, 0))),
                background_sigma_m=np.float64(self.config.background_sigma_m),
                background_clip_percentile=np.float64(self.config.background_clip_percentile),
                seed_snr=np.float64(self.config.seed_snr), grow_snr=np.float64(self.config.grow_snr),
                center_status=self.center_statuses,
                visible_centers_xy_px=(np.column_stack((self.particles['center_x_px'], self.particles['center_y_px']))
                    if self.visible_centers_xy_px is None else self.visible_centers_xy_px),
                full_segmented_area_m2=(np.where(self.particles['touches_edge'], np.nan,
                    self.particles['area_px']*(self.fov_m/max(self.image.shape))**2)
                    if self.full_segmented_area_m2 is None else self.full_segmented_area_m2),
                context_metadata_json=np.asarray(json.dumps(self.context_metadata)),
                coordinate_system=np.asarray("image_axes: x right, y down; origin FoV center"),
            )
        return path


    def save_images(self, directory: str | Path) -> tuple[Path, Path]:
        """Export grayscale HAADF and a circle/center/ID overlay as PNGs.

        PNG intensity is linearly scaled to 8-bit for display; the archive
        retains the original float data. Circle sizes enclose each detected
        region around its centroid and are not particle-size measurements.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        from algorithms.AutoBC import auto_bc
        tone = auto_bc(self.image, self.labels > 0)
        pixels = tone.render(self.image)
        (directory/'auto_bc.json').write_text(json.dumps(tone.metadata(), indent=2), encoding='utf-8')
        raw = Image.fromarray(pixels)
        raw_path = directory / "haadf.png"
        annotated_path = directory / "haadf_annotated.png"
        raw.save(raw_path)
        annotated = raw.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default(size=14)
        for particle in self.particles:
            particle_id = int(particle["id"])
            x, y = float(particle["center_x_px"]), float(particle["center_y_px"])
            radius = float(particle["circle_diameter_px"]) / 2.0
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0), width=2)
            draw.line((x - 4, y, x + 4, y), fill=(0, 255, 255), width=1)
            draw.line((x, y - 4, x, y + 4), fill=(0, 255, 255), width=1)
            text = str(particle_id)
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = min(max(0, x + 6), max(0, raw.width - text_width - 2))
            text_y = min(max(0, y - 18), max(0, raw.height - text_height - 2))
            draw.text((text_x - bbox[0], text_y - bbox[1]), text, font=font,
                      fill=(255, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
        annotated.save(annotated_path)
        return raw_path, annotated_path


def _otsu_threshold(image: np.ndarray) -> float:
    histogram, edges = np.histogram(image, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(histogram, dtype=np.float64)
    moment = np.cumsum(histogram * centers)
    remaining = weight[-1] - weight[:-1]
    valid = (weight[:-1] > 0) & (remaining > 0)
    variance = np.full(255, -np.inf)
    mean_low = moment[:-1][valid] / weight[:-1][valid]
    mean_high = (moment[-1] - moment[:-1][valid]) / remaining[valid]
    variance[valid] = weight[:-1][valid] * remaining[valid] * (mean_low - mean_high) ** 2
    return float(edges[int(np.argmax(variance)) + 1])


def _local_snr_components(smoothed, fov_m, config, otsu_mask):
    """Robust local background and hysteresis, retaining separate bright cores.

    Clip high signals only for background estimation. Local variance includes
    substrate texture; a robust high-pass floor avoids treating noise as signal.
    Watershed grows both strong and weak seeds without joining existing cores.
    """
    oriented = smoothed if config.bright_particles else -smoothed
    if np.median(np.abs(oriented-np.median(oriented))) <= np.finfo(float).eps*max(1., float(np.ptp(oriented))):
        components, count = ndimage.label(otsu_mask, structure=np.ones((3, 3)))
        return components, count, None, None, 'otsu_constant_background'
    pixel_size_m = fov_m/max(smoothed.shape)
    sigma = np.clip(config.background_sigma_m/pixel_size_m, 2., max(2., min(smoothed.shape)/8))
    # Remove illumination slope before clipping strong signals. Otherwise a
    # background ceiling would mistake the bright end of a slope for particles.
    stride = max(1, int(round(sigma/4)))
    yy, xx = np.mgrid[:oriented.shape[0], :oriented.shape[1]]
    xx = xx/max(1, oriented.shape[1]-1)-.5
    yy = yy/max(1, oriented.shape[0]-1)-.5
    design = np.column_stack((np.ones(xx[::stride, ::stride].size),
        xx[::stride, ::stride].ravel(), yy[::stride, ::stride].ravel()))
    samples = oriented[::stride, ::stride].ravel()
    approved = np.ones(len(samples), dtype=bool)
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[approved], samples[approved], rcond=None)[0]
        error = samples-design@coefficients
        median = np.median(error[approved])
        spread = 1.4826*np.median(np.abs(error[approved]-median))
        next_approved = error <= median+3*max(float(spread), np.finfo(float).eps)
        if next_approved.sum() < 3 or np.array_equal(approved, next_approved):
            break
        approved = next_approved
    plane = coefficients[0]+coefficients[1]*xx+coefficients[2]*yy
    detrended = oriented-plane
    capped = np.minimum(detrended, np.percentile(detrended, config.background_clip_percentile))
    # Exact constant-background inputs have no measurable noise scale. Retain
    # established Otsu behavior rather than divide numerical tails by zero.
    if np.ptp(capped) <= np.finfo(float).eps*max(1., float(np.ptp(oriented))):
        components, count = ndimage.label(otsu_mask, structure=np.ones((3, 3)))
        return components, count, None, None, 'otsu_constant_background'
    background = ndimage.gaussian_filter(capped, sigma)
    variance = np.maximum(ndimage.gaussian_filter(capped*capped, sigma)-background*background, 0.)
    highpass = oriented-ndimage.gaussian_filter(oriented, 3.)
    floor = 1.4826*np.median(np.abs(highpass-np.median(highpass)))
    floor = max(float(floor), np.finfo(float).eps*max(1., float(np.ptp(oriented))))
    noise = np.maximum(np.sqrt(variance), floor)
    background += plane
    snr = (oriented-background)/noise
    growth = snr > config.grow_snr
    seeds, seed_count = ndimage.label(snr > config.seed_snr, structure=np.ones((3, 3)))
    strong, strong_count = ndimage.label(otsu_mask, structure=np.ones((3, 3)))
    areas = np.bincount(strong.ravel(), minlength=strong_count+1)
    valid = np.flatnonzero(areas >= config.min_area_px)
    valid = valid[valid != 0]
    seeded_ids = np.unique(strong[snr > config.seed_snr])
    valid = valid[np.isin(valid, seeded_ids)]
    mapping = np.zeros(strong_count+1, dtype=np.int32)
    mapping[valid] = np.arange(1, len(valid)+1)
    markers = mapping[strong]
    growth |= markers > 0
    count = len(valid)
    seed_areas = np.bincount(seeds.ravel(), minlength=seed_count+1)
    seed_regions = ndimage.find_objects(seeds)
    for index, region in enumerate(seed_regions, 1):
        if region is None or seed_areas[index] < 3:
            continue
        local = seeds[region] == index
        if not np.any(markers[region][local]):
            count += 1
            markers[region][local] = count
    markers[~growth] = -1
    cost = np.rint(255*(1-np.clip(snr/(2*config.seed_snr), 0, 1))).astype(np.uint8)
    components = ndimage.watershed_ift(cost, markers.astype(np.int32))
    components = np.where(growth & (components > 0), components, 0).astype(np.int32)
    return components, count, background if config.bright_particles else -background, noise, 'local_snr'


def find_particles_in_image(
    image: npt.ArrayLike,
    *,
    fov_m: float,
    config: FindParticlesConfig = FindParticlesConfig(),
) -> FindParticlesResult:
    """Find bright or dark particles with locally normalized contrast by default.

    FoV is the image's longest-side extent in meters, with square pixels.
    Geometric centroids are sorted top-to-bottom then left-to-right; IDs
    start at 1 and correspond to values in labels (0 is background).
    Watershed retains separate strong cores when weak borders touch. Connected
    particles without distinct seeds can still form one region. Edge regions are retained and flagged
    by default, because their full centers cannot be recovered from this FoV.
    """
    if not np.isfinite(fov_m) or fov_m <= 0:
        raise ValueError("fov_m must be finite and positive")
    if not np.isfinite(config.gaussian_sigma_px) or config.gaussian_sigma_px < 0:
        raise ValueError("gaussian_sigma_px must be finite and nonnegative")
    if isinstance(config.min_area_px, bool) or not isinstance(config.min_area_px, (int, np.integer)) or config.min_area_px < 1:
        raise ValueError("min_area_px must be a positive integer")
    if config.threshold is not None and not np.isfinite(config.threshold):
        raise ValueError("threshold must be finite")
    if config.threshold_method not in ('otsu', 'local_snr'):
        raise ValueError('threshold_method must be otsu or local_snr')
    if isinstance(config.maximum_context_expansions, bool) or not isinstance(config.maximum_context_expansions, (int, np.integer)) or config.maximum_context_expansions < 0:
        raise ValueError('maximum_context_expansions must be a nonnegative integer')
    if not np.isfinite([config.background_sigma_m, config.background_clip_percentile, config.seed_snr, config.grow_snr]).all() or config.background_sigma_m <= 0 or not 50 < config.background_clip_percentile < 100 or not 0 < config.grow_snr < config.seed_snr:
        raise ValueError('Invalid local background/SNR configuration')
    raw = np.asarray(image)
    if raw.ndim != 2 or 0 in raw.shape or np.iscomplexobj(raw):
        raise ValueError("image must be a nonempty real 2-D array")
    image = np.array(raw, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values")
    smoothed = ndimage.gaussian_filter(image.astype(np.float64), config.gaussian_sigma_px)
    threshold = config.threshold
    if threshold is None:
        threshold = float(smoothed.flat[0]) if np.ptp(smoothed) == 0 else _otsu_threshold(smoothed)
    mask = smoothed > threshold if config.bright_particles else smoothed < threshold
    background = noise = None
    method = 'explicit' if config.threshold is not None else 'otsu'
    if config.threshold is None and config.threshold_method == 'local_snr' and np.ptp(smoothed) > 0:
        components, count, background, noise, method = _local_snr_components(smoothed, fov_m, config, mask)
        mask = components > 0
    else:
        components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    areas = np.bincount(components.ravel(), minlength=count + 1)
    edge_ids = set(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])).tolist())
    kept = [i for i in range(1, count + 1)
            if areas[i] >= config.min_area_px and not (config.exclude_edge_particles and i in edge_ids)]
    centers = ndimage.center_of_mass(mask, components, kept) if kept else []
    ordered = sorted(zip(kept, centers), key=lambda item: (item[1][0], item[1][1]))
    particles = np.empty(len(ordered), dtype=PARTICLE_DTYPE)
    mapping = np.zeros(count + 1, dtype=np.int32)
    height, width = image.shape
    pixel_size_m = fov_m / max(height, width)
    regions = ndimage.find_objects(components)
    for particle_id, (component_id, (y, x)) in enumerate(ordered, start=1):
        mapping[component_id] = particle_id
        region = regions[component_id - 1]
        local_y, local_x = np.nonzero(components[region] == component_id)
        distances = np.hypot(local_x + region[1].start - x, local_y + region[0].start - y)
        diameter_px = 2.0 * max(4.0, float(distances.max()) + 1.0)
        particles[particle_id - 1] = (
            particle_id, x, y,
            (x + 0.5 - width / 2) * pixel_size_m,
            (y + 0.5 - height / 2) * pixel_size_m,
            areas[component_id], diameter_px, diameter_px * pixel_size_m,
            component_id in edge_ids,
        )
    return FindParticlesResult(particles, mapping[components], image, float(fov_m), float(threshold), config,
                               background, noise, method)


def find_particles(
    microscope: Microscope,
    *,
    output_path: str | Path = "find_particles.npz",
    config: FindParticlesConfig = FindParticlesConfig(),
) -> FindParticlesResult:
    """Acquire the current FoV once, find its particles, and save the result.

    This action does not change focus or stage position. IDs are local to
    this acquisition; it does not track particles between acquisitions.
    """
    fov_m = microscope.get_fov()
    result = find_particles_in_image(microscope.acquire_haadf(), fov_m=fov_m, config=config)
    output_path = Path(output_path)
    capture_context = getattr(microscope, 'acquire_particle_context', None)
    if config.recover_edge_centers and np.any(result.particles['touches_edge']) and callable(capture_context):
        for attempt in range(1, config.maximum_context_expansions+1):
            check_cancelled()
            if not np.isin(result.center_statuses, ['clipped', 'context_clipped']).any():
                break
            image, context_fov = capture_context(fov_m=fov_m*2**attempt,
                reference_fov_m=fov_m, reference_shape_px=result.image.shape)
            context = find_particles_in_image(image, fov_m=context_fov, config=config)
            result, targets = recover_particle_centers(result, context)
            directory = output_path.parent/(output_path.stem+f'_context_{attempt:02d}')
            directory.mkdir(parents=True, exist_ok=True)
            context.save(directory/'context_particles.npz')
            targets.save(directory/'matched_particles.npz')
            targets.save_images(directory)
            # Mark the original image extent on the larger context view.
            annotated = directory/'haadf_annotated.png'
            with Image.open(annotated) as source:
                picture = source.convert('RGB')
            draw = ImageDraw.Draw(picture)
            ratio = (fov_m/max(result.image.shape))/(context_fov/max(context.image.shape))
            h, w = result.image.shape
            cy, cx = (np.asarray(context.image.shape)-1)/2
            draw.rectangle((cx-w*ratio/2, cy-h*ratio/2, cx+w*ratio/2, cy+h*ratio/2),
                           outline='yellow', width=3)
            draw.text((8, 8), 'Yellow box: original FoV; IDs match original particles',
                      fill='yellow', stroke_width=1, stroke_fill='black')
            picture.save(annotated)
    result.save(output_path)
    return result


def recover_particle_centers(result, context):
    """Match visible masks to complete particles in a larger, concentric image.

    Images must share physical center and scan axes. Overlap voting requires
    at least half the original foreground and a unique one-to-one match.
    IDs and visible areas are preserved; recovered centers may lie outside
    the original image. No out-of-frame shape is guessed from a cropped mask.
    """
    if context.fov_m <= result.fov_m:
        raise ValueError('Particle context must have a larger FoV')
    original_pixel = result.fov_m/max(result.image.shape)
    context_pixel = context.fov_m/max(context.image.shape)
    yy, xx = np.mgrid[:result.image.shape[0], :result.image.shape[1]]
    mapped = ndimage.map_coordinates(context.labels,
        [(yy+.5-result.image.shape[0]/2)*original_pixel/context_pixel+context.image.shape[0]/2-.5,
         (xx+.5-result.image.shape[1]/2)*original_pixel/context_pixel+context.image.shape[1]/2-.5],
        order=0, mode='constant', cval=0)
    matches = {}
    for particle in result.particles:
        check_cancelled()
        selected = mapped[result.labels == particle['id']]
        votes = np.bincount(selected, minlength=len(context.particles)+1)
        votes[0] = 0
        winner = int(votes.argmax())
        ranked = np.sort(votes[1:])
        if winner and votes[winner] >= .5*len(selected) and (len(ranked) < 2 or ranked[-2] < .25*ranked[-1]):
            matches[int(particle['id'])] = winner
    duplicate = {candidate for candidate in matches.values() if list(matches.values()).count(candidate) > 1}
    particles = result.particles.copy()
    statuses = result.center_statuses.astype('<U24').copy()
    visible = (np.column_stack((particles['center_x_px'], particles['center_y_px']))
               if result.visible_centers_xy_px is None else result.visible_centers_xy_px.copy())
    full_area = (np.where(particles['touches_edge'], np.nan, particles['area_px']*original_pixel**2)
                 if result.full_segmented_area_m2 is None else result.full_segmented_area_m2.copy())
    annotated = []
    for index, particle in enumerate(particles):
        candidate = matches.get(int(particle['id']))
        if candidate is not None and candidate not in duplicate:
            complete = context.particles[candidate-1]
            target = complete.copy()
            target['id'] = particle['id']
            annotated.append(target)
        if statuses[index] not in ('clipped', 'context_clipped'):
            continue
        if candidate is None:
            statuses[index] = 'unmatched'
        elif candidate in duplicate:
            statuses[index] = 'ambiguous'
        elif complete['touches_edge']:
            statuses[index] = 'context_clipped'
        else:
            for axis in ('x', 'y'):
                offset = float(complete[f'offset_{axis}_m'])
                particle[f'offset_{axis}_m'] = offset
                size = result.image.shape[1 if axis == 'x' else 0]
                particle[f'center_{axis}_px'] = offset/original_pixel+size/2-.5
            particle['circle_diameter_m'] = complete['circle_diameter_m']
            particle['circle_diameter_px'] = complete['circle_diameter_m']/original_pixel
            full_area[index] = complete['area_px']*context_pixel**2
            statuses[index] = 'recovered'
    metadata = dict(context_fov_m=context.fov_m, context_shape_px=list(context.image.shape),
        original_fov_m=result.fov_m,
        context_pixel_size_m=context_pixel, matching='foreground overlap, unique association',
        original_to_context_id={str(key): value for key, value in matches.items() if value not in duplicate},
        recovered_particle_ids=particles['id'][statuses == 'recovered'].tolist())
    updated = replace(result, particles=particles, center_status=statuses,
        visible_centers_xy_px=visible, full_segmented_area_m2=full_area,
        context_metadata=result.context_metadata+(metadata,))
    label_lookup = np.zeros(len(context.particles)+1, dtype=np.int32)
    for original_id, candidate_id in matches.items():
        if candidate_id not in duplicate:
            label_lookup[candidate_id] = original_id
    targets = replace(context, particles=np.asarray(annotated, dtype=PARTICLE_DTYPE),
                      labels=label_lookup[context.labels])
    return updated, targets
