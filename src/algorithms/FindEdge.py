"""Trace particle outlines and cover them with sparse, ordered square FoVs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from adapters.cancellation import check_cancelled

from adapters.base import Microscope
from algorithms.FindParticles import FindParticlesConfig, find_particles_in_image


BOX_DTYPE = np.dtype([
    ("id", "<i4"), ("center_x_px", "<f8"), ("center_y_px", "<f8"),
    ("offset_x_m", "<f8"), ("offset_y_m", "<f8"),
    ("fov_m", "<f8"), ("extends_outside_image", "?"),
    ("contour_id", "<i4"), ("arc_length_px", "<f8"),
])


@dataclass(frozen=True)
class FindEdgeConfig:
    particles: FindParticlesConfig = field(default_factory=FindParticlesConfig)
    outer_edges_only: bool = True
    grid_phase_steps: int = 4
    center_on_edge: bool = True


@dataclass(frozen=True)
class FindEdgeResult:
    image: np.ndarray
    labels: np.ndarray
    fov_m: float
    box_fov_m: float
    edge_points_px: np.ndarray
    contour_offsets: np.ndarray
    contour_particle_ids: np.ndarray
    contour_touches_frame: np.ndarray
    boxes: np.ndarray

    @property
    def box_centers_m(self):
        """(x, y) image-axis offsets suitable for Zoom2Fit in this view."""
        return np.column_stack((self.boxes["offset_x_m"], self.boxes["offset_y_m"]))

    def save(self, path: str | Path):
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("FindEdge output must have a .npz extension")
        pixel_size = self.fov_m / max(self.image.shape)
        height, width = self.image.shape
        edge_offsets = (self.edge_points_px + 0.5 - np.array([width, height]) / 2) * pixel_size
        with path.open("wb") as stream:
            np.savez_compressed(stream, schema_version=np.int32(2), image=self.image,
                labels=self.labels, fov_m=np.float64(self.fov_m), box_fov_m=np.float64(self.box_fov_m),
                pixel_size_m=np.float64(pixel_size), edge_points_px=self.edge_points_px,
                edge_offsets_m=edge_offsets, contour_offsets=self.contour_offsets,
                contour_particle_ids=self.contour_particle_ids,
                contour_touches_frame=self.contour_touches_frame,
                boxes=self.boxes, box_centers_m=self.box_centers_m,
                coordinate_system=np.asarray("image_axes: x right, y down; origin FoV center"))
        return path

    def save_overlay(self, path: str | Path):
        """Save HAADF with cyan contours, yellow square FoVs and red centers."""
        data = self.image.astype(np.float64)
        low, high = float(data.min()), float(data.max())
        pixels = (np.rint(np.clip((data - low) / (high - low), 0, 1) * 255).astype(np.uint8)
                  if high > low else np.zeros(data.shape, dtype=np.uint8))
        canvas = Image.fromarray(pixels).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        # Pixel boundary coordinates are half-integers, consistent with integer pixel centers.
        for start, stop in zip(self.contour_offsets[:-1], self.contour_offsets[1:]):
            draw.line([tuple(p) for p in self.edge_points_px[start:stop]], fill=(0, 255, 255), width=1)
        side_px = self.box_fov_m / (self.fov_m / max(self.image.shape))
        for box in self.boxes:
            x, y = float(box["center_x_px"]), float(box["center_y_px"])
            half = side_px / 2
            draw.rectangle((x - half, y - half, x + half, y + half), outline=(255, 255, 0), width=1)
            draw.line((x - 1, y, x + 1, y), fill=(255, 0, 0))
            draw.line((x, y - 1, x, y + 1), fill=(255, 0, 0))
            draw.text((x + 2, y + 2), str(int(box['id'])), fill=(255, 255, 255),
                      stroke_width=1, stroke_fill=(0, 0, 0))
        path = Path(path)
        canvas.save(path)
        return path


def _trace_contours(mask):
    """Trace oriented pixel-cell borders; repeat each loop's first point last."""
    padded = np.pad(mask, 1)
    sides = (
        (mask & ~padded[:-2, 1:-1], (0, 0), (1, 0)),
        (mask & ~padded[1:-1, 2:], (1, 0), (1, 1)),
        (mask & ~padded[2:, 1:-1], (1, 1), (0, 1)),
        (mask & ~padded[1:-1, :-2], (0, 1), (0, 0)),
    )
    outgoing = {}
    for exposed, first, second in sides:
        ys, xs = np.nonzero(exposed)
        for x, y in zip(xs.tolist(), ys.tolist()):
            a, b = (x + first[0], y + first[1]), (x + second[0], y + second[1])
            outgoing.setdefault(a, set()).add(b)
    loops = []
    while outgoing:
        start = min(outgoing, key=lambda p: (p[1], p[0]))
        current = start
        points = [start]
        previous_direction = (1, 0)
        while True:
            choices = outgoing[current]
            # Right turn before straight/left prevents crossing at diagonal pixel contacts.
            dx, dy = previous_direction
            preferred = [( -dy, dx), (dx, dy), (dy, -dx), (-dx, -dy)]
            def rank(point):
                return preferred.index((point[0] - current[0], point[1] - current[1]))
            next_point = min(choices, key=rank)
            choices.remove(next_point)
            if not choices:
                del outgoing[current]
            previous_direction = (next_point[0] - current[0], next_point[1] - current[1])
            points.append(next_point)
            current = next_point
            if current == start:
                break
        loops.append(np.asarray(points, dtype=np.float64) - 0.5)
    return loops



def _ordered_edge_centers(loop, side_px):
    """Cover successive contour intervals with maximal spans, trying loop phases.

    A center must lie on this contour and cover every sample in its interval.
    Bounding rectangles also guarantee coverage between samples on pixel borders.
    Phase trials reduce the small residual interval at the closure. This is a
    heuristic, not a globally minimal square covering.
    """
    points = np.empty((2 * (len(loop) - 1), 2))
    points[::2] = loop[:-1]
    points[1::2] = (loop[:-1] + loop[1:]) / 2
    tree = cKDTree(points)
    half = side_px / 2 + 1e-10
    best = None
    for start in np.unique(np.linspace(0, len(points), 9, endpoint=False, dtype=int)):
        check_cancelled()
        ordered = np.roll(points, -start, axis=0)
        ordered = np.vstack((ordered, ordered[0]))
        chosen = []
        first = 0
        while first < len(ordered):
            check_cancelled()
            def feasible(stop):
                segment = ordered[first:stop]
                lower = segment.max(axis=0) - half
                upper = segment.min(axis=0) + half
                if np.any(lower > upper):
                    return np.empty(0, dtype=int)
                midpoint = (lower + upper) / 2
                indices = np.asarray(tree.query_ball_point(midpoint, half, p=np.inf), dtype=int)
                candidates = points[indices]
                return indices[np.all((candidates >= lower) & (candidates <= upper), axis=1)]

            # Feasible rectangles shrink as the interval grows; binary search
            # the longest interval that can share an edge-centered square.
            low, high = first + 1, len(ordered) + 1
            while low + 1 < high:
                middle = (low + high) // 2
                if len(feasible(middle)):
                    low = middle
                else:
                    high = middle
            candidates = points[feasible(low)]
            segment = ordered[first:low]
            midpoint = (segment.min(axis=0) + segment.max(axis=0)) / 2
            center = candidates[np.argmin(np.sum((candidates - midpoint)**2, axis=1))]
            chosen.append(center)
            # Share the endpoint so every segment, including the closing one,
            # is wholly covered by at least one square.
            first = low - 1 if low < len(ordered) else low
        centers = np.asarray(chosen)
        neighbors = tree.query_ball_point(centers, half, p=np.inf)
        # Prune only if entire segments remain covered, not just their endpoints.
        segment_neighbors = []
        for indices in neighbors:
            covered = np.zeros(len(points), dtype=bool)
            covered[indices] = True
            segment_neighbors.append(np.flatnonzero(covered & np.roll(covered, -1)))
        neighbors = segment_neighbors
        counts = np.bincount(np.concatenate(neighbors), minlength=len(points))
        keep = np.ones(len(centers), dtype=bool)
        for index in range(len(centers) - 1, -1, -1):
            covered = neighbors[index]
            if np.all(counts[covered] > 1):
                keep[index] = False
                counts[covered] -= 1
        centers = centers[keep]
        lengths = np.maximum(0, side_px - np.abs(centers[:, None] - centers[None, :]))
        overlap = np.prod(lengths, axis=2)
        np.fill_diagonal(overlap, 0)
        quality = (len(centers), float(overlap.sum()))
        if best is None or quality < best[0]:
            best = (quality, centers)
    return best[1]


def _cover_edges(points, side_px, pixel_size, shape, phase_steps, center_on_edge=False, contours=()):
    """Search grid phases for a sparse, exact-size, zero-area-overlap covering."""
    if not len(points):
        return np.empty(0, dtype=BOX_DTYPE)
    # Each contour is sampled separately by the caller; never bridge distinct loops.
    samples = points
    best = None
    for phase_y in np.arange(phase_steps) * side_px / phase_steps:
        for phase_x in np.arange(phase_steps) * side_px / phase_steps:
            phase = np.array([phase_x, phase_y])
            cells = np.unique(np.floor((samples - phase) / side_px).astype(np.int64), axis=0)
            if best is None or len(cells) < len(best[0]):
                best = (cells, phase)
    cells, phase = best
    centers = (cells + 0.5) * side_px + phase
    centers = centers[np.lexsort((centers[:, 0], centers[:, 1]))]
    if center_on_edge:
        centers = np.concatenate([_ordered_edge_centers(loop, side_px)
                                  for loop in contours])
    boxes = np.empty(len(centers), dtype=BOX_DTYPE)
    height, width = shape
    for i, (x, y) in enumerate(centers, start=1):
        outside = x - side_px / 2 < -0.5 or x + side_px / 2 > width - 0.5 or y - side_px / 2 < -0.5 or y + side_px / 2 > height - 0.5
        boxes[i - 1] = (i, x, y, (x + 0.5 - width / 2) * pixel_size,
                        (y + 0.5 - height / 2) * pixel_size, side_px * pixel_size, outside, 0, 0.0)
    return boxes


def _order_boxes_along_contours(boxes, contours):
    """Number centers by projected arclength, not image rows/columns."""
    if not len(boxes):
        return boxes
    samples, contour_ids, arc_positions = [], [], []
    for identifier, loop in enumerate(contours, start=1):
        lengths = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(lengths)]
        # Interleave vertices and midpoints in traversal order; omit duplicate closing vertex.
        points = np.empty((2 * (len(loop) - 1), 2))
        points[::2] = loop[:-1]
        points[1::2] = (loop[:-1] + loop[1:]) / 2
        positions = np.empty(len(points))
        positions[::2] = arc[:-1]
        positions[1::2] = arc[:-1] + lengths / 2
        samples.append(points)
        contour_ids.extend([identifier] * len(points))
        arc_positions.extend(positions)
    tree = cKDTree(np.concatenate(samples))
    centers = np.column_stack((boxes['center_x_px'], boxes['center_y_px']))
    _, nearest = tree.query(centers)
    boxes['contour_id'] = np.asarray(contour_ids)[nearest]
    boxes['arc_length_px'] = np.asarray(arc_positions)[nearest]
    boxes = boxes[np.lexsort((boxes['arc_length_px'], boxes['contour_id']))]
    boxes['id'] = np.arange(1, len(boxes) + 1)
    return boxes


def find_edge_in_image(image, *, fov_m: float, box_fov_m: float,
                       particle_id: int | None = None, config=FindEdgeConfig()):
    """Trace segmented particle boundaries and return square acquisition centers.

    particle_id refers to FindParticles numbering on THIS image. None includes
    all detected particles. outer_edges_only fills holes before tracing.
    Frame-touching regions are flagged: their full physical edge is unknown.
    FoV describes the longest image side, assuming square pixels. Centers are
    image-axis offsets, not absolute stage coordinates. Boxes may extend beyond
    the original image, and are flagged; they are never shrunk to fit it.
    Grid mode minimizes count among sampled grid phases. Edge-centered mode
    covers maximal successive contour intervals and tries multiple loop starts;
    neither heuristic guarantees a global minimum.
    """
    if not np.isfinite(box_fov_m) or box_fov_m <= 0:
        raise ValueError("box_fov_m must be finite and positive")
    if isinstance(config.grid_phase_steps, bool) or not isinstance(config.grid_phase_steps, int) or config.grid_phase_steps < 1:
        raise ValueError("grid_phase_steps must be a positive integer")
    result = find_particles_in_image(image, fov_m=fov_m, config=config.particles)
    if particle_id is not None and (isinstance(particle_id, bool) or not isinstance(particle_id, (int, np.integer)) or particle_id not in result.particles['id']):
        raise ValueError("particle_id is not present in this image")
    contours, ids, touches = [], [], []
    for particle in result.particles:
        identifier = int(particle['id'])
        if particle_id is not None and identifier != particle_id:
            continue
        mask = result.labels == identifier
        if config.outer_edges_only:
            mask = ndimage.binary_fill_holes(mask)
        for loop in _trace_contours(mask):
            contours.append(loop)
            ids.append(identifier)
            touches.append(bool(particle['touches_edge']))
    offsets = np.r_[0, np.cumsum([len(loop) for loop in contours])].astype(np.int64)
    points = np.concatenate(contours) if contours else np.empty((0, 2), dtype=np.float64)
    pixel_size = fov_m / max(result.image.shape)
    side_px = box_fov_m / pixel_size
    if side_px < 1:
        raise ValueError("box FoV is smaller than one image pixel; acquire a higher-resolution image")
    # Cover each actual segment; do not add artificial lines between separate loops.
    samples = np.concatenate([np.concatenate((loop, (loop[:-1] + loop[1:]) / 2)) for loop in contours]) if contours else points
    boxes = _cover_edges(samples, side_px, pixel_size, result.image.shape, config.grid_phase_steps, config.center_on_edge, contours)
    boxes = _order_boxes_along_contours(boxes, contours)
    return FindEdgeResult(result.image, result.labels, float(fov_m), float(box_fov_m),
                          points, offsets, np.asarray(ids, dtype=np.int32), np.asarray(touches, dtype=bool), boxes)


def find_edge(microscope: Microscope, box_fov_m: float, *, particle_id=None, config=FindEdgeConfig()):
    """Acquire the current FoV once and find its particle-edge acquisition boxes."""
    fov_m = microscope.get_fov()
    return find_edge_in_image(microscope.acquire_haadf(), fov_m=fov_m,
                              box_fov_m=box_fov_m, particle_id=particle_id, config=config)
