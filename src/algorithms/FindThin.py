"""EELS log-ratio thin-area mapping. No simulator thickness is read here."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
from adapters.cancellation import check_cancelled
from algorithms.FindParticles import FindParticlesConfig, find_particles_in_image
from algorithms.AutoBC import auto_bc


def complete_particle_footprint(labels, particle_id):
    """Include segmented cores enclosed by a particle's outer region.

    Fill only already identified foreground, preserving background-labelled
    holes and pores. Nested watershed labels must not remove thick interiors
    from the particle-area denominator.
    """
    labels = np.asarray(labels)
    selected = labels == particle_id
    return selected | (ndimage.binary_fill_holes(selected) & (labels > 0))


@dataclass(frozen=True)
class FindThinConfig:
    image_size_px: int = 128
    pixel_time_us: float = 10.
    defocus_fov_fraction: float = .1
    zlp_center_eV: float = 0.
    zlp_half_width_eV: float = 5.
    thin_threshold: float = 1.
    minimum_total_counts: float = 100.
    minimum_zlp_counts: float = 10.
    dispersion_eV: float = 2.
    energy_min_eV: float = -10.
    energy_max_eV: float = 498.


def _validate(config):
    if isinstance(config.image_size_px, bool) or not isinstance(config.image_size_px, int) or config.image_size_px < 2:
        raise ValueError('image_size_px must be an integer greater than one')
    values = [config.pixel_time_us, config.defocus_fov_fraction, config.zlp_center_eV,
        config.zlp_half_width_eV, config.thin_threshold, config.minimum_total_counts, config.minimum_zlp_counts,
        config.dispersion_eV, config.energy_min_eV, config.energy_max_eV]
    if not np.isfinite(values).all() or min(config.pixel_time_us, config.zlp_half_width_eV,
        config.thin_threshold, config.minimum_total_counts, config.minimum_zlp_counts) <= 0 or config.defocus_fov_fraction < 0:
        raise ValueError('Invalid FindThin configuration')
    if config.dispersion_eV <= 0 or config.energy_min_eV >= config.energy_max_eV or config.energy_min_eV > config.zlp_center_eV-config.zlp_half_width_eV or config.energy_max_eV < config.zlp_center_eV+config.zlp_half_width_eV:
        raise ValueError('Energy window and dispersion must cover the zero-loss interval')


@dataclass(frozen=True)
class FindThinResult:
    t_over_lambda: np.ndarray
    particle_mask: np.ndarray
    thin_mask: np.ndarray
    valid_mask: np.ndarray
    labels: np.ndarray
    total_counts: np.ndarray
    zlp_counts: np.ndarray
    haadf: np.ndarray
    energy_eV: np.ndarray
    fov_m: float
    config: FindThinConfig
    metadata: dict

    def summary(self):
        area = int(self.particle_mask.sum())
        thin = int(self.thin_mask.sum())
        pixel_area = (self.fov_m/max(self.particle_mask.shape))**2
        visible_area = area*pixel_area
        full_area = self.metadata.get('particle_full_area_m2')
        has_full_footprint = full_area is not None and np.isfinite(full_area) and full_area > 0
        denominator = max(visible_area, float(full_area)) if has_full_footprint else visible_area
        outside_area = max(0., denominator-visible_area)
        clipped = bool(np.any(np.r_[self.particle_mask[0], self.particle_mask[-1],
                                   self.particle_mask[:, 0], self.particle_mask[:, -1]]))
        regions = []
        for index, slices in enumerate(ndimage.find_objects(self.labels), 1):
            if slices is None:
                continue
            local = self.labels[slices] == index
            y, x = np.nonzero(local)
            centroid = [float(x.mean()+slices[1].start), float(y.mean()+slices[0].start)]
            nearest = np.argmin((x-x.mean())**2+(y-y.mean())**2)
            position = [float(x[nearest]+slices[1].start), float(y[nearest]+slices[0].start)]
            offset = (np.asarray(position)+.5-np.asarray(self.particle_mask.shape[::-1])/2)*np.sqrt(pixel_area)
            record = dict(id=index, centroid_xy_px=centroid, position_xy_px=position, offset_xy_m=offset.tolist(),
                area_px=int(local.sum()), mean_t_over_lambda=float(self.t_over_lambda[slices][local].mean()))
            if 'image_to_stage_matrix' in self.metadata and 'stage_xy_m' in self.metadata:
                record['center_stage_xy_m'] = (np.asarray(self.metadata['stage_xy_m'])+
                    np.asarray(self.metadata['image_to_stage_matrix'])@offset).tolist()
            regions.append(record)
        return dict(thin_area_percent=100*thin*pixel_area/denominator, particle_area_px=area,
            display_scale=max(1., 512/min(self.particle_mask.shape)),
            particle_area_m2=denominator, visible_particle_area_m2=visible_area,
            thin_area_px=thin, thin_area_m2=thin*pixel_area,
            unquantified_area_percent=100*(int((self.particle_mask & ~self.valid_mask).sum())*pixel_area+outside_area)/denominator,
            area_outside_current_fov_percent=100*outside_area/denominator,
            has_complete_context_footprint=bool(has_full_footprint), particle_touches_edge=clipped,
            denominator=('complete particle footprint measured in context' if has_full_footprint else
                'visible segmented particle projection' if clipped else 'segmented particle projection'),
            formula='t/lambda = ln(total recorded spectral counts / zero-loss counts)',
            thin_rule=f'valid particle pixel and t/lambda < {self.config.thin_threshold}',
            energy_range_eV=[float(self.energy_eV[0]), float(self.energy_eV[-1])],
            regions=regions, config=asdict(self.config), acquisition=self.metadata)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        tone = auto_bc(self.haadf, self.particle_mask)
        summary['auto_bc'] = tone.metadata()
        (directory/'thin_regions.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        np.savez_compressed(directory/'find_thin.npz', schema_version=1,
            t_over_lambda=self.t_over_lambda, particle_mask=self.particle_mask,
            thin_mask=self.thin_mask, valid_mask=self.valid_mask, thin_labels=self.labels,
            total_counts=self.total_counts, zlp_counts=self.zlp_counts,
            haadf=self.haadf, energy_eV=self.energy_eV, fov_m=self.fov_m,
            metadata_json=np.asarray(json.dumps(summary)))
        image = Image.fromarray(tone.render(self.haadf)).convert('RGB')
        image.save(directory/'haadf_autobc.png')
        rgb = np.asarray(image).copy()
        rgb[self.thin_mask] = np.rint(.5*rgb[self.thin_mask]+.5*np.array([0, 255, 80])).astype('uint8')
        boundary = self.particle_mask & ~ndimage.binary_erosion(self.particle_mask)
        rgb[boundary] = [255, 220, 0]
        rgb[self.particle_mask & ~self.valid_mask] = [255, 70, 70]
        labeled = Image.fromarray(rgb)
        scale = summary['display_scale']
        if scale > 1:
            labeled = labeled.resize((round(labeled.width*scale), round(labeled.height*scale)), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(labeled)
        font = ImageFont.load_default(size=14)
        for region in summary['regions']:
            x, y = region['position_xy_px']
            x, y = (x+.5)*scale-.5, (y+.5)*scale-.5
            draw.line((x-4, y, x+4, y), fill='cyan')
            draw.line((x, y-4, x, y+4), fill='cyan')
            draw.text((x+5, y+5), f"Thin #{region['id']}", font=font, fill='white', stroke_width=1, stroke_fill='black')
        text = f"t/lambda < {self.config.thin_threshold:g}: {summary['thin_area_percent']:.2f}% of particle"
        if summary['particle_touches_edge'] and not summary['has_complete_context_footprint']:
            text += '\nClipped particle: visible area only'
        elif summary['area_outside_current_fov_percent'] > 0:
            text += '\nPartial EELS coverage: thin percentage is a lower bound'
        box = draw.multiline_textbbox((8, 8), text, font=font)
        draw.rectangle((0, 0, box[2]+8, box[3]+8), fill='black')
        draw.multiline_text((8, 8), text, font=font, fill='yellow')
        labeled.save(directory/'haadf_thin_annotated.png')
        classification = np.zeros((*self.particle_mask.shape, 3), np.uint8)
        classification[self.particle_mask] = [150, 150, 150]
        classification[self.thin_mask] = [0, 255, 80]
        classification[self.particle_mask & ~self.valid_mask] = [255, 70, 70]
        mask_image = Image.fromarray(classification)
        if scale > 1:
            mask_image = mask_image.resize(labeled.size, Image.Resampling.NEAREST)
        mask_image.save(directory/'thin_mask.png')
        # Green is thin, red is thick; gray is unquantified, black is outside particle.
        value = np.clip(np.nan_to_num(self.t_over_lambda, nan=0.)/2, 0, 1)
        thickness = np.zeros((*value.shape, 3), np.uint8)
        thickness[..., 0] = np.rint(255*value).astype('uint8')
        thickness[..., 1] = np.rint(255*(1-value)).astype('uint8')
        thickness[~self.particle_mask] = 0
        thickness[self.particle_mask & ~self.valid_mask] = 128
        heatmap = Image.fromarray(thickness)
        if scale > 1:
            heatmap = heatmap.resize(labeled.size, Image.Resampling.NEAREST)
        heatmap.save(directory/'t_over_lambda.png')
        return directory/'haadf_thin_annotated.png'


def analyze_thin(spectra, energy_eV, *, particle_mask, fov_m, haadf=None,
                 config=FindThinConfig(), metadata=None):
    """Analyze a y,x,energy count cube in bounded row chunks.

    Particle area is supplied independently of thickness classification so
    vacuum cannot inflate the thin percentage. Invalid spectra count as
    unquantified area, not thin. Finite spectral range can underestimate t/lambda.
    """
    _validate(config)
    spectra = np.asanyarray(spectra)
    energy = np.asarray(energy_eV, float)
    mask = np.asarray(particle_mask, bool)
    if spectra.ndim != 3 or energy.shape != (spectra.shape[-1],) or mask.shape != spectra.shape[:2] or not mask.any():
        raise ValueError('Expected a nonempty particle mask and y,x,energy spectrum image')
    if not np.isfinite(fov_m) or fov_m <= 0 or not np.isfinite(energy).all() or not np.all(np.diff(energy) > 0):
        raise ValueError('Invalid FoV or ascending energy calibration')
    center, width = config.zlp_center_eV, config.zlp_half_width_eV
    zlp = np.abs(energy-center) <= width
    if not zlp.any() or energy[0] > center-width or energy[-1] < center+width:
        raise ValueError('Spectrum must contain the complete zero-loss integration interval')
    shape = spectra.shape[:2]
    total, zero = np.zeros(shape), np.zeros(shape)
    valid = np.zeros(shape, bool)
    negative = 0
    for row in range(0, shape[0], 8):
        check_cancelled()
        block = np.asarray(spectra[row:row+8], dtype=np.float64)
        finite = np.isfinite(block).all(axis=-1)
        negative += int((block < 0).sum())
        block = np.maximum(np.nan_to_num(block, nan=0., posinf=0., neginf=0.), 0.)
        total[row:row+8] = block.sum(axis=-1)
        zero[row:row+8] = block[..., zlp].sum(axis=-1)
        valid[row:row+8] = finite
    valid &= (total >= config.minimum_total_counts) & (zero >= config.minimum_zlp_counts)
    ratio = np.full(shape, np.nan)
    ratio[valid] = np.log(total[valid]/zero[valid])
    thin = mask & valid & (ratio < config.thin_threshold)
    labels, _ = ndimage.label(thin, structure=np.ones((3, 3)))
    image = np.asarray(haadf if haadf is not None else total, dtype=np.float32)
    if image.shape != shape or not np.isfinite(image).all():
        raise ValueError('HAADF must match the spectrum image scan axes')
    info = dict(metadata or {})
    info.update(negative_channel_count=negative,
        total_integration='all recorded energy channels; no power-law subtraction',
        zero_loss_channels_eV=energy[zlp].tolist(),
        finite_window_note='Unrecorded high-energy losses underestimate t/lambda; no model-based correction applied')
    return FindThinResult(ratio, mask.copy(), thin, valid, labels, total, zero,
                          image, energy, float(fov_m), config, info)


def find_thin(microscope, *, output_directory, particle_mask=None, config=FindThinConfig(), progress=None,
              focus_reference_m=None, defocus_offset_m=200e-9):
    """Acquire SI at FoV/10, or relative to a measured autofocus reference."""
    report_progress = progress if progress is not None else lambda message: None
    _validate(config)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    fov = float(microscope.get_fov())
    if not np.isfinite(fov) or fov <= 0:
        raise ValueError('Current FoV must be finite and positive')
    check_cancelled()
    if focus_reference_m is None:
        report_progress('Setting thickness-mapping defocus to FoV/10...')
        microscope.set_effective_defocus(fov*config.defocus_fov_fraction)
    else:
        if not np.isfinite([focus_reference_m, defocus_offset_m]).all():
            raise ValueError('Focus reference and defocus offset must be finite')
        # Measure the footprint at focus: deliberate EELS defocus broadens
        # HAADF edges and must not enlarge the physical-area denominator.
        microscope.set_defocus(float(focus_reference_m))
    # A background-rich overview prevents a close-up's thick core from
    # setting a threshold that erases its weak, thin edges from the denominator.
    context = chosen_context = None
    if particle_mask is None and callable(getattr(microscope, 'acquire_particle_context', None)):
        report_progress('Acquiring wider-view HAADF to identify the complete particle footprint...')
        context_image, context_fov = microscope.acquire_particle_context(fov_m=2*fov,
            reference_fov_m=fov, reference_shape_px=(config.image_size_px, config.image_size_px))
        context = find_particles_in_image(context_image, fov_m=context_fov)
        if not len(context.particles):
            context = find_particles_in_image(context_image, fov_m=context_fov,
                config=FindParticlesConfig(threshold_method='otsu'))
        if len(context.particles):
            chosen_context = min(context.particles, key=lambda p: np.hypot(p['offset_x_m'], p['offset_y_m']))
            context.save(directory/'particle_context.npz')
            context.save_images(directory/'particle_context')
    if focus_reference_m is not None:
        if defocus_offset_m != 0:
            report_progress(f'Setting mapping defocus to autofocus result + {defocus_offset_m*1e9:g} nm...')
        microscope.set_defocus(float(focus_reference_m+defocus_offset_m))
    report_progress(f'Acquiring EELS spectrum image ({config.image_size_px}x{config.image_size_px}, '
        f'{config.pixel_time_us:g} us/pixel) and matching HAADF; this may take tens of seconds...')
    capture = microscope.acquire_eels_spectrum_image(image_size_px=config.image_size_px,
        dwell_time_us=config.pixel_time_us, output_path=None,
        dispersion_eV=config.dispersion_eV, energy_min_eV=config.energy_min_eV, energy_max_eV=config.energy_max_eV)
    report_progress('EELS acquisition complete. Calculating t/lambda and thin-region area...')
    metadata = dict(capture.metadata)
    if particle_mask is None and chosen_context is not None:
        pixel = capture.fov_m/max(capture.haadf.shape)
        context_pixel = context.fov_m/max(context.image.shape)
        yy, xx = np.mgrid[:capture.haadf.shape[0], :capture.haadf.shape[1]]
        complete_mask = complete_particle_footprint(context.labels, chosen_context['id'])
        mapped = ndimage.map_coordinates(complete_mask.astype(np.uint8),
            [(yy+.5-capture.haadf.shape[0]/2)*pixel/context_pixel+context.image.shape[0]/2-.5,
             (xx+.5-capture.haadf.shape[1]/2)*pixel/context_pixel+context.image.shape[1]/2-.5],
            order=0, mode='constant', cval=0)
        selected = mapped > 0
        if selected.any():
            particle_mask = selected
            metadata['particle_mask_source'] = 'background-rich concentric HAADF context, projected to SI pixels'
            metadata['particle_context_segmentation_method'] = context.segmentation_method
            metadata['particle_context_label_id'] = int(chosen_context['id'])
            metadata['particle_context_included_label_ids'] = np.unique(context.labels[complete_mask]).tolist()
            metadata['particle_footprint_method'] = 'selected region plus enclosed segmented foreground; background holes preserved'
            if not chosen_context['touches_edge']:
                metadata['particle_full_area_m2'] = float(complete_mask.sum()*context_pixel**2)
            metadata['particle_context_fov_m'] = context.fov_m
    if particle_mask is None:
        found = find_particles_in_image(capture.haadf, fov_m=capture.fov_m)
        source = 'HAADF local-SNR segmentation'
        if not len(found.particles):
            found = find_particles_in_image(capture.haadf, fov_m=capture.fov_m,
                config=FindParticlesConfig(threshold_method='otsu'))
            source = 'HAADF Otsu fallback'
        if not len(found.particles):
            raise ValueError('No particle footprint identified; supply an aligned particle_mask')
        center = (np.asarray(capture.haadf.shape[::-1])-1)/2
        chosen = min(found.particles, key=lambda p: np.linalg.norm(
            [p['center_x_px']-center[0], p['center_y_px']-center[1]]))
        particle_mask = complete_particle_footprint(found.labels, chosen['id'])
        metadata['particle_mask_source'] = source
        found.save(directory/'particle_footprints.npz')
    elif 'particle_mask_source' not in metadata:
        metadata['particle_mask_source'] = 'caller-supplied aligned mask'
    metadata.update(spectrum_image_file=None, spectrum_image_saved=False)
    if focus_reference_m is None:
        metadata['defocus_mode'] = 'absolute_effective_fov_fraction'
        metadata['requested_effective_defocus_m'] = fov*config.defocus_fov_fraction
    else:
        metadata.update(defocus_mode='relative_to_autofocus',
                        focus_reference_m=float(focus_reference_m), defocus_offset_m=float(defocus_offset_m),
                        requested_lens_defocus_m=float(focus_reference_m+defocus_offset_m),
                        particle_context_lens_defocus_m=float(focus_reference_m))
    result = analyze_thin(capture.spectra, capture.energy_eV, particle_mask=particle_mask,
        fov_m=capture.fov_m, haadf=capture.haadf, config=config, metadata=metadata)
    report_progress('Saving thickness map, annotated HAADF and thin-region positions...')
    result.save(directory)
    report_progress('Thickness mapping complete.')
    return result
