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


def world_to_pixel(header: Mapping[str, Any], ra_deg: float, dec_deg: float) -> tuple[float, float]:
    x, y = WCS(dict(header)).world_to_pixel_values(float(ra_deg), float(dec_deg))
    return float(x), float(y)


# The FITS header keywords carrying the moving target's per-frame ephemeris position, written by
# the scheduler from the object's orbital elements. On LCO MINORPLANET frames the mount tracks the
# object, so these track the WCS field center (CRVAL1/2) frame to frame. RA is sexagesimal hours,
# Dec sexagesimal degrees. Kept as constants so a future keyword change is a one-line edit.
TARGET_RA_HEADER_KEYS = ("CAT-RA",)
TARGET_DEC_HEADER_KEYS = ("CAT-DEC",)


def _first_present_header_value(header: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in header:
            value = header[key]
            if value is not None and str(value).strip():
                return value
    return None


def target_radec_from_header(
    header: Mapping[str, Any],
    *,
    ra_keys: tuple[str, ...] = TARGET_RA_HEADER_KEYS,
    dec_keys: tuple[str, ...] = TARGET_DEC_HEADER_KEYS,
) -> tuple[float, float]:
    """
        Reads a moving target's per-frame RA/Dec (degrees) from a frame header.

        RA is parsed as sexagesimal hours and Dec as sexagesimal degrees (the LCO CAT-RA/CAT-DEC
        convention). Raises ValueError if the keywords are absent or unparseable; callers wrap this
        in their own error type.
    """
    ra_raw = _first_present_header_value(header, ra_keys)
    dec_raw = _first_present_header_value(header, dec_keys)
    if ra_raw is None or dec_raw is None:
        raise ValueError(
            f"Missing moving-target coordinate keywords (looked for RA in {ra_keys}, Dec in {dec_keys})."
        )
    try:
        ra_deg = Angle(str(ra_raw), unit=u.hourangle).to(u.deg).value
        dec_deg = Angle(str(dec_raw), unit=u.deg).to(u.deg).value
    except Exception as exc:
        raise ValueError(f"Unparseable moving-target coordinates: RA={ra_raw!r}, Dec={dec_raw!r}.") from exc
    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg):
        raise ValueError(f"Non-finite moving-target coordinates: RA={ra_raw!r}, Dec={dec_raw!r}.")
    return float(ra_deg), float(dec_deg)


# MJD of 1858-11-17T00:00:00Z, for deriving an MJD from a parsed DATE-OBS when MJD-OBS is absent.
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)


def frame_midpoint_mjd(header: Mapping[str, Any], *, fallback_start: datetime | None = None) -> float:
    """
        MJD (UTC) of a frame's exposure midpoint.

        MJD-OBS and DATE-OBS on LCO frames are the exposure *start* (UTSTART matches DATE-OBS and
        UTSTOP is start + EXPTIME), but a moving target's measured position is where it sat on
        average over the exposure. Interpolating a track at the start time therefore biases every
        predicted position by half an exposure of the object's motion, which is negligible for short
        exposures on a slow mover and arcseconds for long exposures on a fast one.
    """
    if "MJD-OBS" in header:
        start_mjd = float(header["MJD-OBS"])
    elif fallback_start is not None:
        start = fallback_start if fallback_start.tzinfo is not None else fallback_start.replace(tzinfo=timezone.utc)
        start_mjd = (start - _MJD_EPOCH).total_seconds() / 86400.0
    else:
        raise ValueError("Cannot determine an observation time: no MJD-OBS and no fallback start time.")
    if not math.isfinite(start_mjd):
        raise ValueError(f"Non-finite observation time: MJD-OBS={header.get('MJD-OBS')!r}.")
    exposure_seconds = header_float(header, ("EXPTIME",), 0.0)
    if not math.isfinite(exposure_seconds) or exposure_seconds < 0.0:
        exposure_seconds = 0.0
    return start_mjd + exposure_seconds / 2.0 / 86400.0


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
