import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

# Archive headers store the observatory location as OBSGEO-X/Y/Z; wcslib normalizes them to
# OBSGEO-L/B/H on every WCS parse and reports the change as a FITSFixedWarning. The fix is
# purely informational and a WCS is built per frame all over the photometry pipeline, so
# silence the category process-wide.
warnings.filterwarnings('ignore', category=FITSFixedWarning)

MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)
DEFAULT_GAIN = 1.0
DEFAULT_READ_NOISE = 0.0


def world_to_pixel(header: Mapping[str, Any], ra_deg: float, dec_deg: float) -> tuple[float, float]:
    x, y = WCS(dict(header)).world_to_pixel_values(float(ra_deg), float(dec_deg))
    return float(x), float(y)


def target_radec_from_header(header: Mapping[str, Any]) -> tuple[float, float]:
    """
        A moving target's per-frame RA/Dec in degrees, from the LCO ephemeris keywords CAT-RA
        (sexagesimal hours) and CAT-DEC (sexagesimal degrees).
    """
    ra_raw, dec_raw = header.get("CAT-RA"), header.get("CAT-DEC")
    try:
        ra_deg = Angle(str(ra_raw), unit=u.hourangle).to(u.deg).value
        dec_deg = Angle(str(dec_raw), unit=u.deg).to(u.deg).value
    except Exception as exc:
        raise ValueError(f"Cannot read moving-target coordinates: RA={ra_raw!r}, Dec={dec_raw!r}.") from exc
    return float(ra_deg), float(dec_deg)


def frame_midpoint_mjd(header: Mapping[str, Any], *, fallback_start: datetime | None = None) -> float:
    """
        MJD (UTC) of a frame's exposure midpoint.

        MJD-OBS and DATE-OBS are the exposure start, but a moving target's measured position is where
        it sat on average over the exposure, so interpolating a track at the start biases every
        prediction by half an exposure of the object's motion.
    """
    if "MJD-OBS" in header:
        start_mjd = float(header["MJD-OBS"])
    elif fallback_start is not None:
        start = fallback_start if fallback_start.tzinfo is not None else fallback_start.replace(tzinfo=timezone.utc)
        start_mjd = (start - MJD_EPOCH).total_seconds() / 86400.0
    else:
        raise ValueError("Cannot determine an observation time: no MJD-OBS and no fallback start time.")
    # An unusable EXPTIME means no midpoint correction, not a failed frame.
    return start_mjd + max(0.0, optional_float(header.get("EXPTIME"), default=0.0)) / 2.0 / 86400.0


def header_float(header: Mapping[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in header:
            return float(header[key])
    return default


def optional_float(value: Any, default: float = math.nan) -> float:
    """Coerce a possibly missing or malformed value to float, returning default on failure."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


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
        Per-frame WCS, pixel-space aperture geometry, and detector noise parameters, built once
        and reused for the target and every comparison candidate on the frame.

        Constructing a WCS from a header costs on the order of ~10 ms; arcsec_to_pixels and
        world_to_pixel each built one per call, so measuring a frame's candidates rebuilt it
        thousands of times. The plate scale and WCS are frame constants, so they are computed once
        here instead of per candidate.
    """
    wcs: WCS
    aperture_radius_px: float
    annulus_inner_radius_px: float
    annulus_outer_radius_px: float
    gain: float
    read_noise: float

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
        Builds the reusable per-frame geometry: one WCS, the three aperture radii converted to
        pixels via the frame's plate scale, and the detector gain and read noise. Matches
        arcsec_to_pixels/world_to_pixel/frame_gain/frame_read_noise exactly, just without
        re-deriving any of them for every candidate.
    """
    pixel_scale = pixel_scale_arcsec(header)
    return FrameGeometry(
        wcs=WCS(dict(header)),
        aperture_radius_px=float(aperture_radius_arcsec) / pixel_scale,
        annulus_inner_radius_px=float(annulus_inner_radius_arcsec) / pixel_scale,
        annulus_outer_radius_px=float(annulus_outer_radius_arcsec) / pixel_scale,
        gain=frame_gain(header),
        read_noise=frame_read_noise(header),
    )
