from __future__ import annotations

import math
from io import BytesIO
from typing import Any, Sequence

import numpy as np
from fits2image.scaling import calc_zscale_min_max, extract_samples, linear_scale
from PIL import Image, ImageDraw

from datalab.datalab_session.utils.fits_metadata import arcsec_to_pixels, optional_float
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag

COMPARISON_STAR_COLOR = (0, 173, 239)
TARGET_COLOR = (243, 131, 33)
# Diagnostic overlays are resampled so their long side is this many pixels.
OVERLAY_MAX_DIMENSION = 1000


def candidate_overlay_jpeg_bytes(
    *,
    frame: Any,
    image: np.ndarray,
    stars: Sequence[Any],
    measurements: Sequence[Any],
    target_measurement: Any,
    aperture_radius: float,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
) -> bytes:
    """
        Renders the candidate-star overlay for one frame from its full-resolution pixels.

        The target is drawn as the three circles the operation actually measured with — the
        aperture, and the inner and outer bounds of the background annulus in a thinner line.
        Candidates get a single circle at a legible fixed size, since their job is only to mark
        which stars entered the ensemble.

        The image is cropped at full resolution to the region containing every drawn circle plus
        a margin, then resampled so its long side is OVERLAY_MAX_DIMENSION.
    """
    height, width = image.shape
    stars_by_id = {star.candidate_id: star for star in stars}

    comparison_positions: list[tuple[float, float]] = []
    for measurement in measurements:
        if measurement.candidate_id not in stars_by_id:
            continue
        x = float(measurement.x)
        y = float(measurement.y)
        if math.isfinite(x) and math.isfinite(y):
            comparison_positions.append((x, y))

    target_x = float(target_measurement.x)
    target_y = float(target_measurement.y)
    target_position = (target_x, target_y) if math.isfinite(target_x) and math.isfinite(target_y) else None
    positions = comparison_positions + ([target_position] if target_position else [])

    # The target's circles are the operation's real apertures in this frame's pixels; the
    # candidates' shared circle keeps a floor so it stays visible on wide-pixel-scale frames.
    aperture_radius_full = arcsec_to_pixels(frame.header, aperture_radius)
    annulus_inner_radius_full = arcsec_to_pixels(frame.header, annulus_inner_radius)
    annulus_outer_radius_full = arcsec_to_pixels(frame.header, annulus_outer_radius)
    radius_full = max(aperture_radius_full, 14.0)
    # Pad past the largest circle drawn around any position so no annulus is cropped away.
    pad_full = max(2.0 * radius_full, 1.2 * annulus_outer_radius_full)
    x0, y0, x1, y1 = _crop_bounds(positions, pad=pad_full, width=width, height=height)
    crop = image[y0:y1, x0:x1]
    crop_height, crop_width = crop.shape

    scale = OVERLAY_MAX_DIMENSION / max(crop_width, crop_height)
    out_width = max(int(round(crop_width * scale)), 1)
    out_height = max(int(round(crop_height * scale)), 1)
    # Resample the raw flux before stretching: stretching first would clip bright star cores and
    # quantize faint ones to 8 bits before they are averaged, dimming and blurring both. This also
    # keeps the stretch percentiles computed on display-resolution data.
    resampled = Image.fromarray(np.ascontiguousarray(crop, dtype=np.float32)).resize(
        (out_width, out_height), Image.Resampling.LANCZOS
    )
    gray = np.flip(_stretch_to_uint8(np.asarray(resampled)), axis=0)
    overlay = Image.fromarray(np.ascontiguousarray(gray)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    scale_x = out_width / crop_width
    scale_y = out_height / crop_height
    min_dimension = min(out_width, out_height)
    radius = max(radius_full * scale, min_dimension * 0.035, 24.0)
    line_width = max(3, int(round(min_dimension * 0.004)))

    # The image is y-flipped for display, and pixel centers map through a resize as
    # (coordinate + 0.5) * scale - 0.5.
    for x, y in comparison_positions:
        cx = (x - x0 + 0.5) * scale_x - 0.5
        cy = ((crop_height - 1) - (y - y0) + 0.5) * scale_y - 0.5
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=COMPARISON_STAR_COLOR, width=line_width)

    if target_position is not None:
        cx = (target_position[0] - x0 + 0.5) * scale_x - 0.5
        cy = ((crop_height - 1) - (target_position[1] - y0) + 0.5) * scale_y - 0.5
        aperture = aperture_radius_full * scale
        draw.ellipse((cx - aperture, cy - aperture, cx + aperture, cy + aperture), outline=TARGET_COLOR, width=line_width)
        # The annulus circles are rendered at 0.5 or less line width than the aperture radius.
        annulus_line_width = max(1, line_width // 2)
        for annulus_radius_full in (annulus_inner_radius_full, annulus_outer_radius_full):
            annulus = annulus_radius_full * scale
            draw.ellipse(
                (cx - annulus, cy - annulus, cx + annulus, cy + annulus),
                outline=TARGET_COLOR,
                width=annulus_line_width,
            )

    buffer = BytesIO()
    overlay.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def comparison_star_validation_diagnostics(
    *,
    frame: Any,
    stars: Sequence[Any],
    measurements: Sequence[Any],
    frame_zero_point: float,
) -> list[str]:
    diagnostics = [
        (
            "comparison star identifier | RA | Dec | calculated flux | FITS source catalog flux | "
            "calculated magnitude | FITS source catalog magnitude"
        ),
    ]
    stars_by_id = {star.candidate_id: star for star in stars}
    for measurement in measurements:
        star = stars_by_id[measurement.candidate_id]
        catalog_row = star.source_catalog_by_frame.get(frame.fits_path, {})
        calculated_magnitude = _flux_to_magnitude(measurement.net_source_counts, frame_zero_point)
        fits_catalog_flux = optional_float(catalog_row.get("flux"))
        fits_catalog_mag = optional_float(catalog_row.get("mag"))
        diagnostics.append(
            "comparison-star validation row: "
            f"{star.candidate_id} | "
            f"{_format_float(star.ra_deg, precision=4)} | "
            f"{_format_float(star.dec_deg, precision=4)} | "
            f"{_format_float(measurement.net_source_counts, precision=0)} | "
            f"{_format_float(fits_catalog_flux, precision=0)} | "
            f"{_format_float(calculated_magnitude, precision=3)} | "
            f"{_format_float(fits_catalog_mag, precision=3)}"
        )
    return diagnostics


def _stretch_to_uint8(image_data: np.ndarray) -> np.ndarray:
    """
        Stretches image data to uint8 for display with the same fits2image zscale autoscaling
        used for the other datalab JPEGs (see fits2image.scaling.auto_scale): black point at the
        sample median, white point from a zscale fit of the sorted samples, and a 2.5 gamma.
    """
    finite = np.nan_to_num(np.asarray(image_data, dtype=float))
    height, width = finite.shape
    samples = extract_samples(finite, {'NAXIS1': width, 'NAXIS2': height}, nsamples=min(2000, finite.size))
    if samples.size >= 10:
        zmin = float(np.median(samples))
        _, zmax, _ = calc_zscale_min_max(samples, contrast=0.1, iterations=1)
        zmax = float(zmax)
        if not math.isfinite(zmin) or not math.isfinite(zmax) or zmax <= zmin:
            zmin, zmax = float(np.min(finite)), float(np.max(finite))
    else:
        zmin, zmax = float(np.min(finite)), float(np.max(finite))
    return linear_scale(finite, zmin, zmax, max_val=255, gamma_adjust=2.5)


def _crop_bounds(
    positions: Sequence[tuple[float, float]],
    pad: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
        Bounds of the region containing every position plus pad on all sides, clamped to the
        image, in full-resolution pixel coordinates. Falls back to the whole image when there
        are no positions or the clamped region is degenerate.
    """
    if not positions:
        return 0, 0, width, height
    x0 = max(int(math.floor(min(x for x, _ in positions) - pad)), 0)
    y0 = max(int(math.floor(min(y for _, y in positions) - pad)), 0)
    x1 = min(int(math.ceil(max(x for x, _ in positions) + pad)) + 1, width)
    y1 = min(int(math.ceil(max(y for _, y in positions) + pad)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        return 0, 0, width, height
    return x0, y0, x1, y1


def _flux_to_magnitude(flux: float, zero_point: float) -> float:
    if not math.isfinite(flux) or flux <= 0.0 or not math.isfinite(zero_point):
        return math.nan
    magnitude, _ = flux_to_mag(flux, 0.0)
    return float(magnitude) + zero_point


def _format_float(value: float, *, precision: int) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"
