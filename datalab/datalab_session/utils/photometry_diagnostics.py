from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datalab.datalab_session.utils.flux_to_mag import flux_to_mag

COMPARISON_STAR_COLOR = (0, 173, 239)
TARGET_COLOR = (243, 131, 33)


def candidate_overlay_jpeg_base64(
    *,
    frame: Any,
    stars: Sequence[Any],
    measurements: Sequence[Any],
    target_measurement: Any,
    aperture_radius_px: float,
) -> str:
    image = _normalize_image_for_jpeg(frame.image)
    draw = ImageDraw.Draw(image)
    font = _diagnostic_overlay_font(frame.width, frame.height)
    stars_by_id = {star.candidate_id: star for star in stars}
    min_dimension = max(min(frame.width, frame.height), 1)
    radius = max(float(aperture_radius_px), min_dimension * 0.018, 14.0)
    line_width = max(3, int(round(min_dimension * 0.004)))
    label_padding = max(3, int(round(min_dimension * 0.004)))

    for measurement in measurements:
        if measurement.candidate_id not in stars_by_id:
            continue
        x = float(measurement.x)
        y = _display_y(float(measurement.y), frame.height)
        if not math.isfinite(x) or not math.isfinite(y):
            continue

        label = measurement.candidate_id
        halo_bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(halo_bbox, outline=(0, 0, 0), width=line_width + 2)
        draw.ellipse(halo_bbox, outline=COMPARISON_STAR_COLOR, width=line_width)

        label_x = x + radius + label_padding
        label_y = y - radius - label_padding
        label_bbox = draw.textbbox((label_x, label_y), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        if label_x + label_width + label_padding > frame.width:
            label_x = max(x - radius - label_width - label_padding, 0)
        if label_y < 0:
            label_y = min(y + radius + label_padding, max(frame.height - label_height - label_padding, 0))
        draw.text((label_x, label_y), label, fill=COMPARISON_STAR_COLOR, font=font)

    target_x = float(target_measurement.x)
    target_y = _display_y(float(target_measurement.y), frame.height)
    if math.isfinite(target_x) and math.isfinite(target_y):
        target_bbox = (
            target_x - radius,
            target_y - radius,
            target_x + radius,
            target_y + radius,
        )
        draw.ellipse(target_bbox, outline=(0, 0, 0), width=line_width + 2)
        draw.ellipse(target_bbox, outline=TARGET_COLOR, width=line_width)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
        fits_catalog_flux = _optional_float(catalog_row.get("flux"))
        fits_catalog_mag = _optional_float(catalog_row.get("mag"))
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


def _normalize_image_for_jpeg(image_data: np.ndarray) -> Image.Image:
    finite = np.asarray(image_data, dtype=float)
    finite_values = finite[np.isfinite(finite)]
    if finite_values.size:
        zmin, zmax = np.percentile(finite_values, (1, 99.7))
    else:
        zmin, zmax = 0.0, 1.0
    if not math.isfinite(float(zmin)) or not math.isfinite(float(zmax)) or zmax <= zmin:
        zmin = float(np.nanmin(finite_values)) if finite_values.size else 0.0
        zmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    scaled = np.clip((finite - zmin) / (zmax - zmin), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    gray = (scaled * 255.0).astype(np.uint8)
    gray = np.flip(gray, axis=0)
    return Image.fromarray(gray).convert("RGB")


def _display_y(y: float, height: int) -> float:
    return float(height - 1) - y


def _diagnostic_overlay_font(width: int, height: int) -> ImageFont.ImageFont:
    font_size = max(32, min(160, int(round(min(width, height) * 0.05))))
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _flux_to_magnitude(flux: float, zero_point: float) -> float:
    if not math.isfinite(flux) or flux <= 0.0 or not math.isfinite(zero_point):
        return math.nan
    magnitude, _ = flux_to_mag(flux, 0.0)
    return float(magnitude) + zero_point


def _optional_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _format_float(value: float, *, precision: int) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"
