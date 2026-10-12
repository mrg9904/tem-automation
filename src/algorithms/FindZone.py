"""Image-only CIF Kikuchi indexing and symmetry-aware zone-axis suggestions."""
from dataclasses import dataclass, asdict
from itertools import combinations, product
from pathlib import Path
import csv
import json
import math
import numpy as np
from scipy import ndimage, signal, optimize
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
from adapters.cancellation import check_cancelled
from algorithms.crystallography import load_crystal_geometry, electron_wavelength_angstrom, stage_rotation


@dataclass(frozen=True)
class FindZoneConfig:
    analysis_size: int = 256
    angular_steps: int = 360
    rho_bins: int = 1024
    maximum_observed_bands: int = 18
    d_min_angstrom: float = .75
    pair_tolerance_deg: float = 1.5
    match_tolerance_deg: float = .8
    maximum_hypotheses: int = 6000
    template_shortlist: int = 36
    maximum_candidates: int = 4

    def validate(self):
        bounds = dict(analysis_size=(64, 1024), angular_steps=(90, 1440), rho_bins=(128, 4096),
            maximum_observed_bands=(3, 60), maximum_hypotheses=(32, 50000),
            template_shortlist=(1, 128), maximum_candidates=(1, 16))
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError('Invalid FindZone '+name)
        if not np.isfinite([self.d_min_angstrom, self.pair_tolerance_deg, self.match_tolerance_deg]).all() or min(
                self.d_min_angstrom, self.pair_tolerance_deg, self.match_tolerance_deg) <= 0:
            raise ValueError('Invalid spacing or angular tolerance')


def _png(array, path, *, signed=False):
    data = np.asarray(array, float)
    if signed:
        limit = max(float(np.percentile(np.abs(data), 99)), 1e-12)
        pixels = np.clip(.5+.5*data/limit, 0, 1)
    else:
        lo, hi = np.percentile(data, [1, 99])
        pixels = np.clip((data-lo)/max(hi-lo, 1e-12), 0, 1)
    Image.fromarray(np.rint(pixels*255).astype(np.uint8)).save(path)


def preprocess(image, calibration, config):
    image = np.asarray(image, float)
    if image.ndim != 2 or not np.isfinite(image).all() or min(image.shape) < 32:
        raise ValueError('A finite 2-D Ronchigram with at least 32 pixels per axis is required')
    offsets, scales = np.asarray(calibration['offset_yx_rad']), np.asarray(calibration['scale_yx_rad'])
    if offsets.shape != (2,) or scales.shape != (2,) or not np.isfinite([offsets, scales]).all() or np.any(scales == 0):
        raise ValueError('Invalid angular calibration')
    axes = [offset+np.arange(size)*scale for offset, size, scale in zip(offsets, image.shape, scales)]
    for axis in range(2):
        if scales[axis] < 0:
            axes[axis] = axes[axis][::-1]
            image = np.flip(image, axis=axis)
    size = min(config.analysis_size, min(image.shape))
    ty, tx = [np.linspace(np.tan(axis[0]), np.tan(axis[-1]), size) for axis in axes]
    xx, yy = np.meshgrid(tx, ty)
    iy = (np.arctan(yy)-axes[0][0])/(axes[0][1]-axes[0][0])
    ix = (np.arctan(xx)-axes[1][0])/(axes[1][1]-axes[1][0])
    canonical = ndimage.map_coordinates(image, [iy, ix], order=1, mode='nearest')
    smoothed = ndimage.gaussian_filter(canonical, 1.)
    background = ndimage.gaussian_filter(smoothed, 18.)
    mask = background > max(float(np.percentile(background, 90))*.1, 1e-12)
    mask[:6] = mask[-6:] = False; mask[:, :6] = mask[:, -6:] = False
    mask = ndimage.binary_erosion(mask, iterations=3)
    residual = (smoothed-background)/np.maximum(background, 1e-12)
    residual -= np.median(residual[mask]) if mask.any() else 0.
    if mask.sum() < residual.size*.02 or np.std(residual[mask]) < 1e-7:
        raise ValueError('Insufficient illuminated area or Kikuchi contrast')
    return canonical, background, residual, mask, (ty, tx)


def detect_bands(residual, mask, axes, wavelength, crystal, config):
    yy, xx = np.meshgrid(axes[0], axes[1], indexing='ij')
    extent = float(np.hypot(np.abs(axes[0]).max(), np.abs(axes[1]).max()))
    rho = np.linspace(-extent, extent, config.rho_bins)
    profiles = np.zeros((config.angular_steps, config.rho_bins))
    angles = np.arange(config.angular_steps)*np.pi/config.angular_steps
    values = residual[mask]
    x, y = xx[mask], yy[mask]
    possible_widths = wavelength/crystal.d_angstrom
    candidates = []
    for index, angle in enumerate(angles):
        if index % 30 == 0:
            check_cancelled()
        bins = (x*np.cos(angle)+y*np.sin(angle)-rho[0])/(rho[1]-rho[0])
        lo = np.floor(bins).astype(int); fraction = bins-lo
        count = np.bincount(lo, weights=1-fraction, minlength=config.rho_bins)+np.bincount(lo+1, weights=fraction, minlength=config.rho_bins)
        total = np.bincount(lo, weights=values*(1-fraction), minlength=config.rho_bins)+np.bincount(lo+1, weights=values*fraction, minlength=config.rho_bins)
        profile = np.divide(total, count, out=np.zeros_like(total), where=count > 0)[:config.rho_bins]
        profile = ndimage.gaussian_filter1d(profile, 1.2)
        valid = count[:config.rho_bins] > count.max()*.35
        profile[~valid] = 0
        profiles[index] = profile
        noise = max(float(np.median(np.abs(np.diff(profile[valid]))))*.9, 1e-6)
        positive, _ = signal.find_peaks(profile, prominence=3*noise, distance=4)
        negative, _ = signal.find_peaks(-profile, prominence=3*noise, distance=4)
        pairs = []
        for a in positive:
            for b in negative:
                width = abs(rho[a]-rho[b])
                if not .001 < width < min(.07, possible_widths.max()*1.2):
                    continue
                score = np.sqrt(max(0., profile[a]*-profile[b]))
                # Prefer pairs with strong evidence at BOTH opposite edges.
                center = (rho[a]+rho[b])/2
                pairs.append((score, angle, center, width))
        candidates.extend(sorted(pairs, reverse=True)[:12])
    candidates.sort(reverse=True)
    bands = []
    for score, angle, center, width in candidates:
        normal = np.array([np.cos(angle), np.sin(angle), -center])
        normal /= np.linalg.norm(normal)
        if any(np.arccos(np.clip(abs(normal@np.array(band['normal'])), 0, 1)) < np.deg2rad(2.)
               and abs(center-band['rho']) < .002 for band in bands):
            continue
        bands.append(dict(id=len(bands)+1, normal=normal.tolist(), phi_rad=float(angle),
                          rho=float(center), width_rad=float(width), score=float(score)))
        if len(bands) >= config.maximum_observed_bands:
            break
    return bands, profiles, rho, angles


def _basis(a, b):
    second = b-a*(a@b)
    second /= np.linalg.norm(second)
    return np.column_stack((a, second, np.cross(a, second)))


class PatternRenderer:
    """Reusable calibrated rays and reflection data, with bounded chunk arrays."""
    def __init__(self, crystal, axes, wavelength):
        yy, xx = np.meshgrid(axes[0], axes[1], indexing='ij')
        rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        self.rays = rays/np.linalg.norm(rays, axis=-1, keepdims=True)
        self.crystal = crystal
        self.half = wavelength/(2*crystal.d_angstrom)
        self.extent = np.hypot(np.abs(axes[0]).max(), np.abs(axes[1]).max())

    def __call__(self, matrix, sigma_rad=.0015):
        normals = self.crystal.normals @ matrix.T
        visible = np.flatnonzero(np.abs(normals[:, 2]) < self.extent+self.half+4*sigma_rad)
        pattern = np.zeros(self.rays.shape[:2])
        for start in range(0, len(visible), 8):
            indices = visible[start:start+8]
            distance = np.einsum('yxj,ij->iyx', self.rays, normals[indices])
            half = self.half[indices, None, None]
            positive = np.exp(-.5*((distance-half)/sigma_rad)**2)
            negative = np.exp(-.5*((distance+half)/sigma_rad)**2)
            pattern += np.einsum('i,iyx->yx', self.crystal.weights[indices], positive-negative)
        return pattern

    def value_and_jacobian(self, matrix, sigma_rad, rotation_vector):
        """Analytic derivatives for left SO(3) perturbation and log line width."""
        omega = np.asarray(rotation_vector)
        theta = np.linalg.norm(omega)
        skew = np.array([[0., -omega[2], omega[1]], [omega[2], 0., -omega[0]], [-omega[1], omega[0], 0.]])
        a = (1-np.cos(theta))/theta**2 if theta > 1e-5 else .5-theta**2/24
        b = (theta-np.sin(theta))/theta**3 if theta > 1e-5 else 1/6-theta**2/120
        left_jacobian = np.eye(3)+a*skew+b*(skew@skew)
        normals = self.crystal.normals@matrix.T
        visible = np.flatnonzero(abs(normals[:, 2]) < self.extent+self.half+4*sigma_rad)
        value = np.zeros(self.rays.shape[:2])
        jacobian = np.zeros((*value.shape, 4))
        for start in range(0, len(visible), 8):
            indices = visible[start:start+8]
            normal = normals[indices]
            distance = np.einsum('yxj,ij->iyx', self.rays, normal)
            half, weight = self.half[indices, None, None], self.crystal.weights[indices, None, None]
            positive_delta, negative_delta = distance-half, distance+half
            positive = np.exp(-.5*(positive_delta/sigma_rad)**2)
            negative = np.exp(-.5*(negative_delta/sigma_rad)**2)
            value += np.sum(weight*(positive-negative), axis=0)
            distance_derivative = np.einsum('iyxk,kj->iyxj',
                np.cross(normal[:, None, None, :], self.rays[None]), left_jacobian)
            derivative = weight*(-positive_delta*positive+negative_delta*negative)/sigma_rad**2
            jacobian[..., :3] += np.einsum('iyx,iyxj->yxj', derivative, distance_derivative)
            jacobian[..., 3] += np.sum(weight*(positive_delta**2*positive-negative_delta**2*negative)/sigma_rad**2, axis=0)
        return value, jacobian


def render_pattern(matrix, crystal, axes, wavelength, sigma_rad=.0015):
    return PatternRenderer(crystal, axes, wavelength)(matrix, sigma_rad)


def equivalent_to_any(matrix, previous, symmetries, tolerance_rad):
    if not len(previous):
        return False
    equivalents = np.einsum('ij,sjk->sik', matrix, np.asarray(symmetries))
    traces = np.einsum('sij,pij->sp', equivalents, np.asarray(previous))
    return bool(np.any((traces-1)/2 > np.cos(tolerance_rad)))


def _standardize(image, mask):
    image = image-ndimage.gaussian_filter(image, max(3., image.shape[0]/14))
    values = image[mask]
    return (image-values.mean())/max(float(values.std()), 1e-12)


def _standardize_with_jacobian(image, jacobian, mask):
    sigma = max(3., image.shape[0]/14)
    image = image-ndimage.gaussian_filter(image, sigma)
    jacobian = jacobian-ndimage.gaussian_filter(jacobian, (sigma, sigma, 0))
    centered = image-image[mask].mean()
    derivative = jacobian-jacobian[mask].mean(axis=0)
    std = max(float(centered[mask].std()), 1e-12)
    std_derivative = np.mean(centered[mask, None]*derivative[mask], axis=0)/std
    return centered/std, derivative/std-centered[..., None]*std_derivative/std**2


def index_bands(bands, residual, mask, axes, crystal, wavelength, config, progress):
    if len(bands) < 3:
        return [], []
    observed = np.asarray([band['normal'] for band in bands])
    # Collinear harmonic reflections are a single plane-normal constraint.
    seed_indices = []
    for index, normal in enumerate(crystal.normals):
        if not any(abs(normal@crystal.normals[other]) > .99999 for other in seed_indices):
            seed_indices.append(index)
        if len(seed_indices) >= 90:
            break
    normal_pairs = np.array(list(combinations(seed_indices, 2)))
    pair_angles = np.arccos(np.clip(np.abs(np.sum(crystal.normals[normal_pairs[:, 0]]*
                                  crystal.normals[normal_pairs[:, 1]], axis=1)), 0, 1))
    hypotheses = []
    for first, second in combinations(range(min(7, len(bands))), 2):
        angle = np.arccos(np.clip(abs(observed[first]@observed[second]), 0, 1))
        if angle < np.deg2rad(20):
            continue
        matches = np.flatnonzero(abs(pair_angles-angle) < np.deg2rad(config.pair_tolerance_deg))
        matches = matches[np.argsort(abs(pair_angles[matches]-angle))]
        for match_index, match in enumerate(matches[:200]):
            if match_index % 20 == 0:
                check_cancelled()
            for a, b in (normal_pairs[match], normal_pairs[match][::-1]):
                for sign_a, sign_b in product((-1, 1), repeat=2):
                    lab = _basis(sign_a*observed[first], sign_b*observed[second])
                    matrix = lab @ _basis(crystal.normals[a], crystal.normals[b]).T
                    dots = abs(observed @ matrix @ crystal.normals.T)
                    errors = np.arccos(np.clip(dots.max(axis=1), 0, 1))
                    quality = float(np.exp(-.5*(errors/np.deg2rad(config.match_tolerance_deg))**2).sum())
                    hypotheses.append((quality, matrix))
        check_cancelled()
        if len(hypotheses) >= config.maximum_hypotheses:
            break
    hypotheses.sort(key=lambda item: -item[0])
    # Keep inequivalent orientations; retain multiple ambiguous candidates.
    shortlist = []
    for quality, matrix in hypotheses:
        if equivalent_to_any(matrix, [previous[1] for previous in shortlist], crystal.symmetries, np.deg2rad(2.)):
            continue
        shortlist.append((quality, matrix))
        if len(shortlist) >= config.template_shortlist:
            break
    progress(f'Testing {len(shortlist)} orientation hypotheses against the image...')
    small_size = 96
    small_axes = tuple(np.linspace(axis[0], axis[-1], small_size) for axis in axes)
    small_image = ndimage.zoom(residual, small_size/residual.shape[0], order=1)
    small_mask = ndimage.zoom(mask.astype(float), small_size/mask.shape[0], order=0) > .5
    observed_image = _standardize(small_image, small_mask)
    renderer = PatternRenderer(crystal, small_axes, wavelength)
    candidates = []
    for template_index, (quality, base_matrix) in enumerate(shortlist):
        if template_index % 6 == 0:
            progress(f'Comparing CIF templates {template_index+1}/{len(shortlist)}...')
        check_cancelled()
        scores = []
        # Symmetry-equivalent indexing has the same geometric bands, while
        # illustrative bright/dark polarity may vary by representative.
        for symmetry in crystal.symmetries:
            matrix = base_matrix @ symmetry
            for sigma in (.0007, .0015, .003):
                template = _standardize(renderer(matrix, sigma), small_mask)
                correlation = float(np.mean(template[small_mask]*observed_image[small_mask]))
                scores.append((abs(correlation), sigma, np.sign(correlation) or 1., matrix))
        score, sigma, polarity, matrix = max(scores, key=lambda item: item[0])
        candidates.append(dict(matrix=matrix, band_vote_score=quality, correlation=score, sigma_rad=sigma, polarity=polarity))
    candidates.sort(key=lambda item: -item['correlation'])
    initial_scores = [dict(band_vote_score=item['band_vote_score'], correlation=item['correlation'],
                          orientation_matrix=item['matrix'].tolist()) for item in candidates]
    refined = []
    to_refine = candidates[:max(8, config.maximum_candidates)]
    for fit_index, candidate in enumerate(to_refine):
        progress(f'Refining orientation candidate {fit_index+1}/{len(to_refine)}...')
        check_cancelled()
        base = candidate['matrix']
        cached_parameters = cached_residual = cached_jacobian = None
        def evaluate(parameters):
            nonlocal cached_parameters, cached_residual, cached_jacobian
            check_cancelled()
            if cached_parameters is not None and np.array_equal(parameters, cached_parameters):
                return cached_residual, cached_jacobian
            matrix = Rotation.from_rotvec(parameters[:3]).as_matrix() @ base
            template, jacobian = renderer.value_and_jacobian(matrix, np.exp(parameters[3]), parameters[:3])
            template, jacobian = _standardize_with_jacobian(template, jacobian, small_mask)
            cached_parameters = parameters.copy()
            cached_residual = (candidate['polarity']*template-observed_image)[small_mask]
            cached_jacobian = candidate['polarity']*jacobian[small_mask]
            return cached_residual, cached_jacobian
        fit = optimize.least_squares(lambda parameters: evaluate(parameters)[0], [0., 0., 0., np.log(candidate['sigma_rad'])],
            jac=lambda parameters: evaluate(parameters)[1],
            bounds=([-.045]*3+[np.log(.0003)], [.045]*3+[np.log(.008)]), max_nfev=50,
            ftol=1e-5, xtol=1e-5, diff_step=1e-4)
        matrix = Rotation.from_rotvec(fit.x[:3]).as_matrix() @ base
        template = _standardize(renderer(matrix, np.exp(fit.x[3])), small_mask)
        correlation = abs(float(np.mean(template[small_mask]*observed_image[small_mask])))
        refined.append(dict(candidate, matrix=matrix, sigma_rad=float(np.exp(fit.x[3])), correlation=correlation))
    refined.sort(key=lambda item: -item['correlation'])
    return refined[:config.maximum_candidates], initial_scores


def zone_suggestions(matrix, crystal, current_tilt=(0., 0.), candidate_id='A'):
    """Solve absolute lab R_y(TY) @ R_x(TX), including Laue-equivalent labels."""
    tx0, ty0 = current_tilt
    body = stage_rotation(tx0, ty0).T @ matrix
    rows = []
    for equivalent, symmetry in enumerate(crystal.symmetries):
        for axis in product((0, 1), repeat=3):
            if axis == (0, 0, 0):
                continue
            direction = crystal.cell @ axis
            direction /= np.linalg.norm(direction)
            vector = body @ symmetry @ direction
            options = []
            for sign in (-1, 1):
                v = sign*vector
                tx = np.arctan2(v[1], v[2])
                ty = np.arctan2(-v[0], np.hypot(v[1], v[2]))
                for add_tx, add_ty in ((0., 0.), (np.pi, np.pi)):
                    target = np.array([tx+add_tx, ty+add_ty])
                    target = np.array([tx0, ty0])+(target-np.array([tx0, ty0])+np.pi)%(2*np.pi)-np.pi
                    aligned = stage_rotation(*target) @ vector
                    if abs(aligned[2]) < 1-1e-8:
                        continue
                    delta = target-np.array([tx0, ty0])
                    options.append((np.linalg.norm(delta), target, delta))
            _, target, delta = min(options, key=lambda item: item[0])
            current_vector = matrix @ symmetry @ direction
            angle = float(np.rad2deg(np.arccos(np.clip(abs(current_vector[2]), 0, 1))))
            labels = '['+''.join(map(str, axis))+']'
            same = next((row for row in rows if np.linalg.norm(np.array(row['target_tilt_deg'])-np.rad2deg(target)) < .1), None)
            if same is not None:
                if labels not in same['possible_zone_axes']:
                    same['possible_zone_axes'].append(labels)
                continue
            rows.append(dict(candidate_id=candidate_id, possible_zone_axes=[labels],
                beam_axis_angle_deg=angle, delta_tx_deg=float(np.rad2deg(delta[0])),
                delta_ty_deg=float(np.rad2deg(delta[1])), target_tilt_deg=np.rad2deg(target).tolist(),
                equivalent_id=equivalent, tilt_limits_checked=False))
    return sorted(rows, key=lambda row: row['beam_axis_angle_deg'])


def band_assignments(bands, matrix, crystal, wavelength, tolerance_deg):
    normals = crystal.normals @ matrix.T
    assignments = []
    for band in bands:
        errors = np.rad2deg(np.arccos(np.clip(abs(normals @ np.asarray(band['normal'])), 0, 1)))
        near = np.flatnonzero(errors <= errors.min()+.03)
        widths = wavelength/crystal.d_angstrom[near]/np.maximum(np.linalg.norm(normals[near, :2], axis=1), 1e-12)
        selected = near[np.argmin(abs(np.log(widths/band['width_rad'])))]
        assignments.append(dict(detected_band_id=band['id'], hkl=crystal.hkl[selected].tolist(),
            angular_error_deg=float(errors[selected]), matched=bool(errors[selected] <= tolerance_deg),
            harmonic_or_close_plane_hypotheses=crystal.hkl[near].tolist()))
    return assignments


def nearest_small_index_zones(matrix, crystal, *, max_index=3, count=5):
    """Identify nearby primitive zones, including signed indices beyond binary targets."""
    candidates = []
    for indices in product(range(-max_index, max_index+1), repeat=3):
        if indices == (0, 0, 0) or next(value for value in indices if value) < 0:
            continue
        if math.gcd(math.gcd(abs(indices[0]), abs(indices[1])), abs(indices[2])) != 1:
            continue
        direction = crystal.cell @ indices
        direction /= np.linalg.norm(direction)
        lab = matrix @ direction
        angle = np.rad2deg(np.arccos(np.clip(abs(lab[2]), 0, 1)))
        candidates.append((float(angle), indices))
    candidates.sort()
    result = []
    seen = set()
    inverse = np.linalg.inv(crystal.cell)
    for angle, indices in candidates:
        family = []
        for symmetry in crystal.symmetries:
            equivalent = np.rint(inverse @ symmetry @ crystal.cell @ indices).astype(int)
            divisor = math.gcd(math.gcd(abs(int(equivalent[0])), abs(int(equivalent[1]))), abs(int(equivalent[2])))
            equivalent //= max(divisor, 1)
            if next(value for value in equivalent if value) < 0:
                equivalent = -equivalent
            family.append(tuple(map(int, equivalent)))
        family = sorted(set(family))
        signature = tuple(family)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(dict(zone_axis=list(indices), equivalent_zone_axes=[list(axis) for axis in family],
                           beam_axis_angle_deg=angle, max_index_searched=max_index))
        if len(result) >= count:
            break
    return result


def _band_overlay(base, bands, axes, path, *, crystal=None, matrix=None, wavelength=None, indexed_hkl=None):
    picture = base.convert('RGB').copy()
    draw = ImageDraw.Draw(picture)
    xaxis, yaxis = axes[1], axes[0]
    records = bands
    if crystal is not None:
        records = []
        for indices, normal, spacing in zip(crystal.hkl, crystal.normals @ matrix.T, crystal.d_angstrom):
            if indexed_hkl is not None and tuple(indices) not in indexed_hkl:
                continue
            records.append(dict(normal=normal, label='('+','.join(map(str, indices))+')'))
    for index, band in enumerate(records):
        normal = np.asarray(band['normal'])
        points = []
        for x in (xaxis[0], xaxis[-1]):
            if abs(normal[1]) > 1e-10:
                y = -(normal[0]*x+normal[2])/normal[1]
                if yaxis[0] <= y <= yaxis[-1]: points.append((x, y))
        for y in (yaxis[0], yaxis[-1]):
            if abs(normal[0]) > 1e-10:
                x = -(normal[1]*y+normal[2])/normal[0]
                if xaxis[0] <= x <= xaxis[-1]: points.append((x, y))
        if len(points) >= 2:
            pixels = [((x-xaxis[0])/(xaxis[-1]-xaxis[0])*(picture.width-1),
                       (y-yaxis[0])/(yaxis[-1]-yaxis[0])*(picture.height-1)) for x, y in points[:2]]
            draw.line(pixels, fill='cyan' if crystal is None else 'lime', width=1)
            if index < 18:
                px, py = pixels[0]
                draw.text((max(2, min(px, picture.width-65)), max(2, min(py, picture.height-14))),
                    band.get('label') or str(band.get('id', index+1)), fill='yellow', font=ImageFont.load_default(size=11),
                    stroke_width=1, stroke_fill='black')
    picture.save(path)


def find_zone_in_image(image, *, cif_path, calibration, output_directory, config=FindZoneConfig(), progress=print):
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    config.validate()
    progress('Reading CIF and preparing Ronchigram background separation...')
    crystal = load_crystal_geometry(str(Path(cif_path).resolve()), config.d_min_angstrom)
    wavelength = electron_wavelength_angstrom(calibration['voltage_v'])
    _png(image, directory/'01_ronchigram.png')
    canonical, background, residual, mask, axes = preprocess(image, calibration, config)
    _png(canonical, directory/'02_canonical_detector.png')
    _png(background, directory/'03_background.png')
    _png(residual, directory/'04_band_contrast.png', signed=True)
    Image.fromarray(mask.astype(np.uint8)*255).save(directory/'05_analysis_mask.png')
    progress('Detecting Kikuchi band pairs with angular Radon projections...')
    bands, profiles, rho, angles = detect_bands(residual, mask, axes, wavelength, crystal, config)
    _png(profiles, directory/'06_radon.png', signed=True)
    with Image.open(directory/'02_canonical_detector.png') as base:
        _band_overlay(base, bands, axes, directory/'07_detected_bands.png')
    (directory/'detected_bands.json').write_text(json.dumps(bands, indent=2), encoding='utf-8')
    progress(f'Detected {len(bands)} band hypotheses. Indexing against CIF plane geometry...')
    candidates, initial = index_bands(bands, residual, mask, axes, crystal, wavelength, config, progress)
    cutoff = max(.6, candidates[0]['correlation']-.12) if candidates else 1.
    plausible_count = sum(candidate['correlation'] >= cutoff for candidate in candidates)
    report = dict(status=('unindexed' if not candidates else 'tentative' if plausible_count == 0 else
                          'ambiguous' if plausible_count > 1 else 'indexed'),
        calibration=calibration, cif_path=crystal.source, space_group=crystal.space_group,
        cell_angstrom=crystal.cell.tolist(), config=asdict(config), detected_bands=bands,
        symmetry_count=len(crystal.symmetries), candidates=[], tilt_suggestions=[],
        conventions='R maps crystal Cartesian to laboratory; canonical axes ascend; Stage R_y(TY) @ R_x(TX)',
        limitations=['scores are image correlations, not probabilities',
                    'plausibility cutoff is an initial heuristic, not calibrated confidence',
                    'tilt recommendations have not been applied or checked against hardware limits',
                    'Friedel/Laue-equivalent solutions do not identify absolute polarity'])
    (directory/'orientation_hypotheses.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
    for index, candidate in enumerate(candidates):
        name = chr(ord('A')+index)
        matrix = candidate['matrix']
        beam_direction = np.linalg.solve(crystal.cell, matrix.T @ [0., 0., 1.])
        beam_direction /= max(np.max(abs(beam_direction)), 1e-12)
        report['candidates'].append(dict(id=name, orientation_matrix=matrix.tolist(),
            beam_direction_uvw_proportional=beam_direction.tolist(), correlation=candidate['correlation'],
            fitted_line_sigma_rad=candidate['sigma_rad'], band_vote_score=candidate['band_vote_score'],
            plausible=candidate['correlation'] >= cutoff))
        report['candidates'][-1]['nearest_small_index_zones'] = nearest_small_index_zones(matrix, crystal)
        assignments = band_assignments(bands, matrix, crystal, wavelength, config.match_tolerance_deg)
        report['candidates'][-1]['band_assignments'] = assignments
        (directory/f'indexed_bands_{name}.json').write_text(json.dumps(assignments, indent=2), encoding='utf-8')
        template = render_pattern(matrix, crystal, axes, wavelength, candidate['sigma_rad'])
        _png(template, directory/f'08_candidate_{name}_pattern.png', signed=True)
        _png(_standardize(template, mask), directory/f'candidate_{name}_normalized.png', signed=True)
        with Image.open(directory/'04_band_contrast.png') as observed, \
             Image.open(directory/f'candidate_{name}_normalized.png') as predicted:
            comparison = Image.new('RGB', (observed.width*2, observed.height+24), 'black')
            comparison.paste(observed, (0, 24)); comparison.paste(predicted, (observed.width, 24))
            draw = ImageDraw.Draw(comparison)
            draw.text((5, 4), 'Observed band contrast', fill='white')
            draw.text((observed.width+5, 4), f'CIF model {name}: score {candidate["correlation"]:.3f}', fill='white')
            comparison.save(directory/f'10_candidate_{name}_comparison.png')
        with Image.open(directory/'02_canonical_detector.png') as base:
            _band_overlay(base, [], axes, directory/f'09_candidate_{name}_indexed.png',
                          crystal=crystal, matrix=matrix, wavelength=wavelength,
                          indexed_hkl={tuple(item['hkl']) for item in assignments if item['matched']})
        rows = zone_suggestions(matrix, crystal, calibration.get('stage_tilt_xy_rad', (0., 0.)), name)
        if candidate['correlation'] >= cutoff:
            report['tilt_suggestions'].extend(rows)
    (directory/'find_zone.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (directory/'zone_tilts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate', 'possible_zone_axes', 'beam_axis_angle_deg', 'delta_TX_deg', 'delta_TY_deg', 'target_TX_deg', 'target_TY_deg'])
        for row in report['tilt_suggestions']:
            writer.writerow([row['candidate_id'], '/'.join(row['possible_zone_axes']), row['beam_axis_angle_deg'],
                row['delta_tx_deg'], row['delta_ty_deg'], *row['target_tilt_deg']])
    rows = sorted(report['tilt_suggestions'], key=lambda row: row['beam_axis_angle_deg'])[:12]
    table = Image.new('RGB', (920, 60+26*len(rows)), 'white')
    draw = ImageDraw.Draw(table)
    font = ImageFont.load_default(size=14)
    columns = [8, 90, 350, 475, 605, 745]
    for x, label in zip(columns, ['Candidate', 'Possible axes', 'Axis angle', 'Delta TX', 'Delta TY', 'Target TX/TY']):
        draw.text((x, 8), label, fill='black', font=font)
    for index, row in enumerate(rows):
        values = [row['candidate_id'], '/'.join(row['possible_zone_axes']), f"{row['beam_axis_angle_deg']:.2f} deg",
                  f"{row['delta_tx_deg']:+.2f} deg", f"{row['delta_ty_deg']:+.2f} deg",
                  '/'.join(f'{value:+.2f}' for value in row['target_tilt_deg'])]
        for x, value in zip(columns, values):
            draw.text((x, 34+26*index), value, fill='black', font=font)
    table.save(directory/'11_zone_tilts.png')
    np.savez_compressed(directory/'find_zone.npz', image=image, canonical_image=canonical,
        band_contrast=residual, analysis_mask=mask, radon=profiles, rho=rho, phi_rad=angles,
        orientations=np.asarray([item['matrix'] for item in candidates]).reshape(-1, 3, 3),
        metadata_json=np.asarray(json.dumps(report)))
    progress('Saving orientation candidates, indexed overlays and zone-axis tilt table...')
    return report
