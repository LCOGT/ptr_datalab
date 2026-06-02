import math
from typing import Any, Mapping, Sequence

from astropy.wcs import WCS


def build_celestial_wcs(
    header: Mapping[str, Any],
    *,
    error_class: type[Exception] = ValueError,
) -> WCS:
    wcs = WCS(dict(header))
    if not wcs.has_celestial:
        raise error_class("Header does not contain a usable celestial WCS.")
    return wcs


def world_to_pixel(
    header: Mapping[str, Any],
    ra_deg: float,
    dec_deg: float,
    *,
    error_class: type[Exception] = ValueError,
) -> tuple[float, float]:
    wcs = build_celestial_wcs(header, error_class=error_class)
    x, y = wcs.world_to_pixel_values(float(ra_deg), float(dec_deg))
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise error_class("WCS world-to-pixel conversion produced non-finite coordinates.")
    return float(x), float(y)


def pixel_to_world(
    header: Mapping[str, Any],
    x: float,
    y: float,
    *,
    error_class: type[Exception] = ValueError,
) -> tuple[float, float]:
    wcs = build_celestial_wcs(header, error_class=error_class)
    ra, dec = wcs.pixel_to_world_values(float(x), float(y))
    if not math.isfinite(float(ra)) or not math.isfinite(float(dec)):
        raise error_class("WCS pixel-to-world conversion produced non-finite coordinates.")
    return float(ra), float(dec)


def header_float(header: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in header:
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                return default
            if math.isfinite(value):
                return value
    return default
