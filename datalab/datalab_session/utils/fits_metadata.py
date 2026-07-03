import math
from typing import Any, Mapping

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def world_to_pixel(header: Mapping[str, Any], ra_deg: float, dec_deg: float) -> tuple[float, float]:
    x, y = WCS(dict(header)).world_to_pixel_values(float(ra_deg), float(dec_deg))
    return float(x), float(y)


def header_float(header: Mapping[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in header:
            return float(header[key])
    return default


def optional_float(value: Any) -> float:
    """Coerce a (possibly missing/malformed) catalog or header value to float, NaN on failure."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def pixel_scale_arcsec(header: Mapping[str, Any]) -> float:
    """
        Arcsec-per-pixel for a frame, from its WCS astrometric solution (preferred) or the
        nominal PIXSCALE header, so an angular aperture can be sized in pixels per frame.
    """
    try:
        scale = float(np.mean(proj_plane_pixel_scales(WCS(dict(header)).celestial) * 3600.0))
        if math.isfinite(scale) and scale > 0.0:
            return scale
    except Exception:
        pass
    if "PIXSCALE" in header:
        pixscale = float(header["PIXSCALE"])
        if math.isfinite(pixscale) and pixscale > 0.0:
            return pixscale
    raise ValueError("Cannot determine a pixel scale for an arcsec-unit aperture.")


def aperture_unit_scale(header: Mapping[str, Any], aperture_unit: str) -> float:
    """
        Divisor that converts aperture radii to pixels for a frame: 1.0 when radii are already
        pixels, or the frame's arcsec/px plate scale when they are an angular size (so the
        aperture covers the same sky on every telescope/instrument regardless of pixel scale).
    """
    if aperture_unit == "px":
        return 1.0
    if aperture_unit != "arcsec":
        raise ValueError(f"Unsupported aperture_unit {aperture_unit!r}; expected 'px' or 'arcsec'.")
    return pixel_scale_arcsec(header)
