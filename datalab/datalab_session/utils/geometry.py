import math
from typing import Any, Mapping, Sequence

import numpy as np


def distance_pixels(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def too_close_to_target(
    *,
    candidate_xy: tuple[float, float],
    target_xy: tuple[float, float],
    annulus_outer_radius_px: float,
    aperture_radius_px: float,
    target_proximity_factor: float,
) -> bool:
    return distance_pixels(*candidate_xy, *target_xy) <= max(
        target_proximity_factor * aperture_radius_px,
        annulus_outer_radius_px,
    )


def too_close_to_edges(
    x: float,
    y: float,
    width: int,
    height: int,
    annulus_outer_radius_px: float,
    edge_margin_px: float,
) -> bool:
    return (
        x - annulus_outer_radius_px < edge_margin_px
        or y - annulus_outer_radius_px < edge_margin_px
        or x + annulus_outer_radius_px >= width - edge_margin_px
        or y + annulus_outer_radius_px >= height - edge_margin_px
    )


def minimum_neighbor_distances_arcsec(ra_deg: Sequence[float], dec_deg: Sequence[float]) -> np.ndarray:
    """
        For each position, the angular distance to its nearest other position, in arcseconds. One
        vectorised sweep per position; an N x N matrix would be hundreds of MB at a dense field's
        candidate count.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if ra.size < 2:
        return np.full(ra.size, math.inf)
    nearest = np.empty(ra.size, dtype=float)
    for index in range(ra.size):
        separations = angular_distances_arcsec(float(ra[index]), float(dec[index]), ra, dec)
        separations[index] = math.inf
        nearest[index] = separations.min()
    return nearest


def angular_distance_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    cos_angle = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return math.degrees(math.acos(cos_angle)) * 3600.0


def angular_distances_arcsec(
    ra_deg: float,
    dec_deg: float,
    other_ra_deg: np.ndarray,
    other_dec_deg: np.ndarray,
) -> np.ndarray:
    """Angular distance from one position to an array of positions, in arcseconds."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    other_ra = np.radians(other_ra_deg)
    other_dec = np.radians(other_dec_deg)
    cos_angle = np.sin(dec) * np.sin(other_dec) + math.cos(dec) * np.cos(other_dec) * np.cos(ra - other_ra)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))) * 3600.0


def unit_vectors(ra_deg: Any, dec_deg: Any) -> np.ndarray:
    """Cartesian unit vectors for one or many RA/Dec positions, shaped (..., 3)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)
