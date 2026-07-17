import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

# Archive headers store the observatory location as OBSGEO-X/Y/Z; wcslib normalizes them to
# OBSGEO-L/B/H on every WCS parse and reports the change as a FITSFixedWarning. The fix is
# purely informational and a WCS is built per frame all over the photometry pipeline, so
# silence the category process-wide.
warnings.filterwarnings('ignore', category=FITSFixedWarning)


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


DEFAULT_GAIN = 1.0
DEFAULT_READ_NOISE = 0.0


def frame_gain(header: Mapping[str, Any]) -> float:
    """Detector gain (e-/ADU) from the frame header, falling back to DEFAULT_GAIN."""
    return header_float(header, ("GAIN", "EGAIN"), DEFAULT_GAIN)


def frame_read_noise(header: Mapping[str, Any]) -> float:
    """Detector read noise (e-) from the frame header, falling back to DEFAULT_READ_NOISE."""
    return header_float(header, ("RDNOISE", "READNOIS", "READNOISE"), DEFAULT_READ_NOISE)


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


def arcsec_to_pixels(header: Mapping[str, Any], angular_radius_arcsec: float) -> float:
    """
        Convert an angular aperture radius to pixels for one frame, using the frame's plate scale.
    """
    return float(angular_radius_arcsec) / pixel_scale_arcsec(header)


@dataclass(frozen=True)
class FrameGeometry:
    """
        Per-frame WCS and pixel-space aperture geometry, built once and reused for the target and
        every comparison candidate on the frame.

        Constructing a WCS from a header costs on the order of ~10 ms; arcsec_to_pixels and
        world_to_pixel each built one per call, so measuring a frame's candidates rebuilt it
        thousands of times. The plate scale and WCS are frame constants, so they are computed once
        here instead of per candidate.
    """
    wcs: WCS
    aperture_radius_px: float
    annulus_inner_radius_px: float
    annulus_outer_radius_px: float

    def world_to_pixel(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        """Pixel coordinates of a sky position using the cached WCS (no header re-parse)."""
        x, y = self.wcs.world_to_pixel_values(float(ra_deg), float(dec_deg))
        return float(x), float(y)


def frame_geometry(
    header: Mapping[str, Any],
    aperture_radius_arcsec: float,
    annulus_inner_radius_arcsec: float,
    annulus_outer_radius_arcsec: float,
) -> FrameGeometry:
    """
        Builds the reusable per-frame geometry: one WCS plus the three aperture radii converted to
        pixels via the frame's plate scale. Matches arcsec_to_pixels/world_to_pixel exactly, just
        without rebuilding the WCS for every candidate.
    """
    pixel_scale = pixel_scale_arcsec(header)
    return FrameGeometry(
        wcs=WCS(dict(header)),
        aperture_radius_px=float(aperture_radius_arcsec) / pixel_scale,
        annulus_inner_radius_px=float(annulus_inner_radius_arcsec) / pixel_scale,
        annulus_outer_radius_px=float(annulus_outer_radius_arcsec) / pixel_scale,
    )
