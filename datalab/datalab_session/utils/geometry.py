import math
from typing import Any, Mapping, Sequence


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


def minimum_angular_neighbor_distance_arcsec(
    cluster: Mapping[str, Any],
    clusters: Sequence[Mapping[str, Any]],
) -> float:
    distances = []
    for other in clusters:
        if other is cluster:
            continue
        distances.append(
            angular_distance_arcsec(cluster["ra_deg"], cluster["dec_deg"], other["ra_deg"], other["dec_deg"])
        )
    return min(distances) if distances else math.inf


def angular_distance_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    cos_angle = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return math.degrees(math.acos(cos_angle)) * 3600.0
