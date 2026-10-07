"""Index separated particles in a HAADF field of view and save NumPy arrays.

Coordinates use pixel centers (x=column, y=row). Offsets are image-axis
coordinates relative to the FoV center, not absolute stage coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
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
    ("touches_edge", "?"),
])


@dataclass(frozen=True)
class IndexParticlesConfig:
    """Threshold applies after smoothing; None selects an Otsu threshold."""

    gaussian_sigma_px: float = 1.0
    min_area_px: int = 16
    threshold: float | None = None
    bright_particles: bool = True
    exclude_edge_particles: bool = False


@dataclass(frozen=True)
class IndexParticlesResult:
    particles: np.ndarray
    labels: npt.NDArray[np.int32]
    image: npt.NDArray[np.float32]
    fov_m: float
    threshold: float
    config: IndexParticlesConfig

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
                schema_version=np.int32(1),
                particles=self.particles,
                labels=self.labels,
                image=self.image,
                image_shape_px=np.asarray(self.image.shape, dtype=np.int64),
                fov_m=np.float64(self.fov_m),
                pixel_size_m=np.float64(self.fov_m / max(self.image.shape)),
                threshold=np.float64(self.threshold),
                gaussian_sigma_px=np.float64(self.config.gaussian_sigma_px),
                min_area_px=np.int64(self.config.min_area_px),
                bright_particles=np.bool_(self.config.bright_particles),
                exclude_edge_particles=np.bool_(self.config.exclude_edge_particles),
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
        low, high = float(self.image.min()), float(self.image.max())
        if high > low:
            pixels = np.rint(np.clip((self.image.astype(np.float64) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        else:
            pixels = np.zeros(self.image.shape, dtype=np.uint8)
        raw = Image.fromarray(pixels)
        raw_path = directory / "haadf.png"
        annotated_path = directory / "haadf_indexed.png"
        raw.save(raw_path)
        annotated = raw.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default(size=14)
        regions = ndimage.find_objects(self.labels)
        for particle in self.particles:
            particle_id = int(particle["id"])
            x, y = float(particle["center_x_px"]), float(particle["center_y_px"])
            region = regions[particle_id - 1]
            local_y, local_x = np.nonzero(self.labels[region] == particle_id)
            distances = np.hypot(local_x + region[1].start - x, local_y + region[0].start - y)
            radius = max(4.0, float(distances.max()) + 1.0)
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


def index_particles_in_image(
    image: npt.ArrayLike,
    *,
    fov_m: float,
    config: IndexParticlesConfig = IndexParticlesConfig(),
) -> IndexParticlesResult:
    """Segment particles using Gaussian smoothing and 8-connected regions.

    FoV is the image's longest-side extent in meters, with square pixels.
    Geometric centroids are sorted top-to-bottom then left-to-right; IDs
    start at 1 and correspond to values in labels (0 is background).
    Touching particles form one region. Edge regions are retained and flagged
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
    for particle_id, (component_id, (y, x)) in enumerate(ordered, start=1):
        mapping[component_id] = particle_id
        particles[particle_id - 1] = (
            particle_id, x, y,
            (x + 0.5 - width / 2) * pixel_size_m,
            (y + 0.5 - height / 2) * pixel_size_m,
            areas[component_id], component_id in edge_ids,
        )
    return IndexParticlesResult(particles, mapping[components], image, float(fov_m), float(threshold), config)


def index_particles(
    microscope: Microscope,
    *,
    output_path: str | Path = "index_particles.npz",
    config: IndexParticlesConfig = IndexParticlesConfig(),
) -> IndexParticlesResult:
    """Acquire the current FoV once, index its particles, and save the result.

    This action does not change focus or stage position. IDs are local to
    this acquisition; it does not track particles between acquisitions.
    """
    fov_m = microscope.get_fov()
    result = index_particles_in_image(microscope.acquire_haadf(), fov_m=fov_m, config=config)
    result.save(output_path)
    return result
