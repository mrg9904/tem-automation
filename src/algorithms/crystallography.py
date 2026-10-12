"""CIF crystal geometry independent of the microscope and simulator."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import gemmi
import numpy as np
from scipy import constants


@dataclass(frozen=True)
class CrystalGeometry:
    cell: np.ndarray
    hkl: np.ndarray
    normals: np.ndarray
    d_angstrom: np.ndarray
    weights: np.ndarray
    symmetries: tuple
    space_group: str
    source: str


@lru_cache(maxsize=8)
def load_crystal_geometry(cif_path, d_min_angstrom=.75):
    if not np.isfinite(d_min_angstrom) or d_min_angstrom <= 0:
        raise ValueError('Minimum d spacing must be positive')
    for block in gemmi.cif.read_file(str(Path(cif_path).resolve())):
        if block.find_value('_cell_length_a') and block.find_values('_atom_site_fract_x'):
            crystal = gemmi.make_small_structure_from_block(block)
            break
    else:
        raise ValueError('CIF contains no cell and atomic sites')
    if crystal.cell.volume <= 0:
        raise ValueError('Invalid CIF cell')
    cell = np.asarray(crystal.cell.orth.mat)
    reciprocal = np.linalg.inv(cell).T
    sites = crystal.get_all_unit_cell_sites()
    fractional = np.asarray([tuple(site.fract) for site in sites])
    limits = np.ceil(np.linalg.norm(cell, axis=0)/d_min_angstrom).astype(int)
    hkl, normals, spacing, strength = [], [], [], []
    for indices in product(*(range(-int(limit), int(limit)+1) for limit in limits)):
        if indices == (0, 0, 0) or next(value for value in indices if value) < 0:
            continue
        g = reciprocal @ indices
        length = np.linalg.norm(g)
        if length > 1/d_min_angstrom+1e-10:
            continue
        g2 = length**2
        factors = np.asarray([site.occ*site.element.c4322.calculate_sf(g2/4)*
            np.exp(-2*np.pi**2*max(0., site.u_iso)*g2) for site in sites])
        intensity = abs(np.sum(factors*np.exp(2j*np.pi*(fractional @ indices))))**2
        hkl.append(indices); normals.append(g/length); spacing.append(1/length); strength.append(intensity)
    strength = np.asarray(strength)
    keep = strength > strength.max()*1e-5
    order = np.flatnonzero(keep)[np.argsort(-strength[keep], kind='stable')]
    symmetry = [np.eye(3)]
    group = gemmi.find_spacegroup_by_name(crystal.spacegroup_hm)
    if group is not None:
        for operation in group.operations():
            rotation = cell @ (np.asarray(operation.rot)/operation.DEN) @ np.linalg.inv(cell)
            if np.linalg.det(rotation) < 0:
                rotation = -rotation  # Friedel-equivalent proper Laue rotation
            if not any(np.allclose(rotation, item, atol=1e-8) for item in symmetry):
                symmetry.append(rotation)
    return CrystalGeometry(cell, np.asarray(hkl)[order], np.asarray(normals)[order],
        np.asarray(spacing)[order], np.sqrt(strength[order]/strength.max()), tuple(symmetry),
        crystal.spacegroup_hm, str(Path(cif_path).resolve()))


def electron_wavelength_angstrom(voltage_v):
    if not np.isfinite(voltage_v) or voltage_v <= 0:
        raise ValueError('Voltage must be finite and positive')
    energy = constants.e*voltage_v
    return constants.h/np.sqrt(2*constants.m_e*energy*(1+energy/(2*constants.m_e*constants.c**2)))*1e10


def stage_rotation(tx, ty):
    cx, sx, cy, sy = np.cos(tx), np.sin(tx), np.cos(ty), np.sin(ty)
    return np.array([[cy, sy*sx, sy*cx], [0., cx, -sx], [-sy, cy*sx, cy*cx]])
