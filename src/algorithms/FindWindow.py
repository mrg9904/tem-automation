"""Locate a complete square opening using image evidence only.

Coordinates are pixel centers: x right, y down. A square's orientation is
ambiguous modulo 90 degrees; angle_deg is clockwise in display axes, [0, 90).
No mesh, grid pitch, particle positions or known orientation are consulted.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize, spatial
from PIL import Image, ImageDraw, ImageFont
from adapters.cancellation import check_cancelled


@dataclass(frozen=True)
class FindWindowConfig:
    smoothing_sigma_px: float = 2.
    min_side_fraction: float = .08
    max_relative_boundary_error: float = .025
    min_square_fill: float = .85


@dataclass(frozen=True)
class FindWindowResult:
    image: np.ndarray
    center_xy_px: np.ndarray
    corners_xy_px: np.ndarray
    side_px: float
    angle_deg: float
    boundary_error_px: float
    square_fill: float
    candidate_count: int

    def summary(self):
        return dict(center_xy_px=self.center_xy_px.tolist(), side_px=self.side_px,
            angle_deg=self.angle_deg, corners_xy_px=self.corners_xy_px.tolist(),
            boundary_error_px=self.boundary_error_px, square_fill=self.square_fill,
            candidate_count=self.candidate_count,
            angle_convention='clockwise from display +x toward +y, modulo 90 degrees',
            selection='complete square closest to image center; not a grid index identification')

    def save(self, directory, *, anchor=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        if anchor is not None:
            summary['anchor'] = anchor
        (directory/'window.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        np.savez_compressed(directory/'find_window.npz', schema_version=1,
            image=self.image, center_xy_px=self.center_xy_px,
            corners_xy_px=self.corners_xy_px, side_px=self.side_px,
            angle_deg=self.angle_deg, boundary_error_px=self.boundary_error_px,
            metadata_json=np.asarray(json.dumps(summary)))
        lo, hi = np.percentile(self.image, [1, 99])
        raw = Image.fromarray(np.rint(np.clip((self.image-lo)/max(hi-lo, 1e-12), 0, 1)*255).astype('uint8'))
        raw.save(directory/'ronchigram.png')
        labeled = raw.convert('RGB')
        draw = ImageDraw.Draw(labeled)
        points = [tuple(p) for p in self.corners_xy_px]
        draw.line(points+[points[0]], fill=(0, 255, 100), width=3)
        for i, point in enumerate(points):
            draw.text(point, f' C{i+1}', fill=(255, 200, 0), stroke_width=1, stroke_fill='black')
        x, y = self.center_xy_px
        draw.line((x-10, y, x+10, y), fill='cyan', width=2)
        draw.line((x, y-10, x, y+10), fill='cyan', width=2)
        text = f'Central window | side {self.side_px:.1f} px | angle {self.angle_deg:.2f} deg (mod 90)\nCenter x={x:.1f}, y={y:.1f} px | boundary RMS {self.boundary_error_px:.2f} px'
        if anchor is not None:
            text += f"\nSide {anchor['mean_side_m']*1e6:.2f} um | detector angle {anchor['detector_angle_deg']:.2f} deg"
        font = ImageFont.load_default(size=max(12, min(self.image.shape)//55))
        bbox = draw.multiline_textbbox((8, 8), text, font=font)
        draw.rectangle((0, 0, bbox[2]+8, bbox[3]+8), fill='black')
        draw.multiline_text((8, 8), text, font=font, fill='yellow')
        if anchor is not None:
            origin = np.array([100., self.image.shape[0]-80.])
            scales = np.asarray(anchor['angular_scale_yx_rad'])
            for label, direction in (('Detector +x', [np.sign(scales[1]), 0]),
                                     ('Detector +y', [0, np.sign(scales[0])])):
                direction = np.asarray(direction)
                end = origin+direction*50
                normal = np.array([-direction[1], direction[0]])
                draw.line([tuple(origin), tuple(end)], fill='cyan', width=2)
                draw.line([tuple(end-direction*8+normal*4), tuple(end),
                           tuple(end-direction*8-normal*4)], fill='cyan', width=2)
                draw.text(tuple(end+np.array([4, 4])), label, font=font,
                          fill='cyan', stroke_width=1, stroke_fill='black')
        labeled.save(directory/'ronchigram_window.png')
        return directory/'ronchigram_window.png'


def _threshold(image):
    histogram, bins = np.histogram(image, 256)
    p = histogram.astype(float)/image.size
    weight = np.cumsum(p)
    mean = np.cumsum(p*(bins[:-1]+bins[1:])/2)
    variance = (mean[-1]*weight-mean)**2/np.maximum(weight*(1-weight), 1e-15)
    index = np.argmax(variance[:-1])
    return (bins[index]+bins[index+1])/2


def _fit_square(component, config):
    component = ndimage.binary_fill_holes(component)
    boundary = component & ~ndimage.binary_erosion(component)
    yy, xx = np.nonzero(boundary)
    points = np.column_stack((xx, yy)).astype(float)
    hull = spatial.ConvexHull(points)
    vertices = points[hull.vertices]
    edges = np.roll(vertices, -1, axis=0)-vertices
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi/2))
    best = None
    for theta in angles:
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local = points @ rotation
        low, high = local.min(axis=0), local.max(axis=0)
        size = high-low
        area = np.prod(size)
        if best is None or area < best[0]:
            best = area, (low+high)/2 @ rotation.T, size, theta
    _, center, size, theta = best
    if min(size)/max(size) < .85 or min(size) < config.min_side_fraction*min(component.shape):
        return None
    def residual(parameters):
        cx, cy, side, angle = parameters
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = (points-[cx, cy]) @ rotation
        return np.max(np.abs(local), axis=1)-side/2
    fit = optimize.least_squares(residual, [*center, np.mean(size), theta], loss='soft_l1', f_scale=1.)
    cx, cy, side, theta = fit.x
    error = float(np.sqrt(np.mean(residual(fit.x)**2)))
    fill = float(component.sum()/side**2)
    if error/side > config.max_relative_boundary_error or not config.min_square_fill <= fill <= 1.1:
        return None
    theta %= np.pi/2
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*side/2 @ rotation.T+[cx, cy]
    if (corners < -1).any() or (corners[:, 0] > component.shape[1]).any() or (corners[:, 1] > component.shape[0]).any():
        return None
    return np.array([cx, cy]), corners, float(side), float(np.degrees(theta)), error, fill


def find_window_in_image(image, *, config=None):
    """Find the nearest complete square to image center; fail if none is supported.

    Both contrast polarities are tested. Filled holes suppress internal particles;
    contour fitting estimates four sides jointly. Clipped openings are rejected.
    """
    config = config or FindWindowConfig()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 32 or not np.isfinite(image).all():
        raise ValueError('Expected a finite 2-D Ronchigram of at least 32 pixels per axis')
    values = (config.smoothing_sigma_px, config.min_side_fraction, config.max_relative_boundary_error, config.min_square_fill)
    if not np.isfinite(values).all() or config.smoothing_sigma_px < 0 or not 0 < config.min_side_fraction < 1 or config.max_relative_boundary_error <= 0 or not 0 < config.min_square_fill <= 1:
        raise ValueError('Invalid FindWindow configuration')
    if np.ptp(image) <= 0:
        raise ValueError('No image contrast; cannot locate a square window')
    smooth = ndimage.gaussian_filter(image, config.smoothing_sigma_px)
    threshold = _threshold(smooth)
    candidates = []
    for mask in (smooth > threshold, smooth < threshold):
        labels, _ = ndimage.label(mask)
        for label, slices in enumerate(ndimage.find_objects(labels), 1):
            check_cancelled()
            if slices is None or min(s.stop-s.start for s in slices) < config.min_side_fraction*min(image.shape):
                continue
            if any(s.start == 0 or s.stop == image.shape[i] for i, s in enumerate(slices)):
                continue
            fitted = _fit_square(labels == label, config)
            if fitted is not None:
                candidates.append(fitted)
    if not candidates:
        raise ValueError('No complete square window found; include all four grid edges in the Ronchigram')
    target = (np.array(image.shape[::-1])-1)/2
    best = min(candidates, key=lambda c: np.linalg.norm(c[0]-target))
    return FindWindowResult(image.copy(), *best, len(candidates))


def make_window_anchor(result, stage_positions_m, angular_offset_rad, angular_scale_rad):
    """Package externally calibrated geometry for later scans.

    stage_positions_m contains fitted center followed by C1..C4. The device
    supplies ray-to-stage conversion; the algorithm never imports its backend.
    Detector angles use calibrated +x/+y rather than screen axes. A bilinear
    stage patch is provided for normalized window coordinates u,v in [0,1].
    """
    stages = np.asarray(stage_positions_m, dtype=float)
    offset, scale = np.asarray(angular_offset_rad), np.asarray(angular_scale_rad)
    if stages.shape != (5, 2) or offset.shape != (2,) or scale.shape != (2,) or not np.isfinite(np.r_[stages.ravel(), offset, scale]).all() or (scale == 0).any():
        raise ValueError('Expected five finite stage points and two angular calibrations')
    corners = stages[1:]
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0)-corners, axis=1)
    detector = result.corners_xy_px*scale[::-1]+offset[::-1]
    vector = detector[1]-detector[0]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 90)
    return dict(center_stage_xy_m=stages[0].tolist(), corners_stage_xy_m=corners.tolist(),
        side_lengths_m=lengths.tolist(), mean_side_m=float(lengths.mean()),
        corners_detector_xy_rad=detector.tolist(), detector_angle_deg=angle,
        angular_offset_yx_rad=offset.tolist(), angular_scale_yx_rad=scale.tolist(),
        detector_angle_convention='from calibrated detector +x toward +y, modulo 90 degrees',
        stage_patch_origin_m=corners[0].tolist(),
        stage_patch_u_m=(corners[1]-corners[0]).tolist(),
        stage_patch_v_m=(corners[3]-corners[0]).tolist(),
        stage_patch_uv_m=(corners[2]-corners[1]-corners[3]+corners[0]).tolist(),
        center_patch_error_m=float(np.linalg.norm(stages[0]-corners.mean(axis=0))),
        mapping='stage(u,v)=origin+u*U+v*V+u*v*UV; assumes sufficiently weak interior ray distortion',
        units='meters; stage targets use the acquisition device coordinate convention')
