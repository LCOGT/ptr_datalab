from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datalab.datalab_session.utils.fits_metadata import arcsec_to_pixels, optional_float
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag

COMPARISON_STAR_COLOR = (0, 173, 239)
TARGET_COLOR = (243, 131, 33)
# Diagnostic overlays are rendered on a downsampled preview no larger than this on either side.
PREVIEW_MAX_DIMENSION = 2000


@dataclass(frozen=True)
class FramePreview:
    """
        Downsampled display rendering of a frame, captured while the frame's pixels were loaded.

        gray is uint8, display-oriented (y-flipped), block-mean downsampled so that preview
        coordinates = full-resolution coordinates * scale. Diagnostic overlays are drawn on this
        preview, so full-resolution pixels never have to be reloaded or retained for rendering.
    """
    gray: np.ndarray
    scale: float

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])


def build_frame_preview(image: np.ndarray, max_dimension: int = PREVIEW_MAX_DIMENSION) -> FramePreview:
    """
        Builds the downsampled uint8 display preview for a frame.

        Blocks are mean-combined (point sampling would drop stars smaller than the sampling step),
        and the display stretch is computed on the downsampled array, so the only full-resolution
        temporary is the trimmed copy the block reshape may make.
    """
    height, width = image.shape
    step = max(1, math.ceil(max(height, width) / max_dimension))
    if step > 1:
        trimmed = image[: (height // step) * step, : (width // step) * step]
        blocks = trimmed.reshape(height // step, step, width // step, step)
        small = blocks.mean(axis=(1, 3), dtype=np.float64)
    else:
        small = np.asarray(image, dtype=float)
    gray = np.flip(_stretch_to_uint8(small), axis=0)
    return FramePreview(gray=np.ascontiguousarray(gray), scale=1.0 / step)


def candidate_overlay_jpeg_base64(
    *,
    frame: Any,
    preview: FramePreview,
    stars: Sequence[Any],
    measurements: Sequence[Any],
    target_measurement: Any,
    aperture_radius: float,
) -> str:
    image = Image.fromarray(preview.gray).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _diagnostic_overlay_font(preview.width, preview.height)
    stars_by_id = {star.candidate_id: star for star in stars}
    min_dimension = max(min(preview.width, preview.height), 1)
    aperture_radius_px = arcsec_to_pixels(frame.header, aperture_radius) * preview.scale
    radius = max(aperture_radius_px, min_dimension * 0.018, 14.0)
    line_width = max(3, int(round(min_dimension * 0.004)))
    label_padding = max(3, int(round(min_dimension * 0.004)))

    for measurement in measurements:
        if measurement.candidate_id not in stars_by_id:
            continue
        x = float(measurement.x) * preview.scale
        y = _display_y(float(measurement.y) * preview.scale, preview.height)
        if not math.isfinite(x) or not math.isfinite(y):
            continue

        label = _overlay_label(measurement.candidate_id)
        halo_bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(halo_bbox, outline=(0, 0, 0), width=line_width + 2)
        draw.ellipse(halo_bbox, outline=COMPARISON_STAR_COLOR, width=line_width)

        label_x = x + radius + label_padding
        label_y = y - radius - label_padding
        label_bbox = draw.textbbox((label_x, label_y), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        if label_x + label_width + label_padding > preview.width:
            label_x = max(x - radius - label_width - label_padding, 0)
        if label_y < 0:
            label_y = min(y + radius + label_padding, max(preview.height - label_height - label_padding, 0))
        draw.text(
            (label_x, label_y),
            label,
            fill=COMPARISON_STAR_COLOR,
            font=font,
        )

    target_x = float(target_measurement.x) * preview.scale
    target_y = _display_y(float(target_measurement.y) * preview.scale, preview.height)
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
    return (scaled * 255.0).astype(np.uint8)


def _display_y(y: float, height: int) -> float:
    return float(height - 1) - y


def _diagnostic_overlay_font(width: int, height: int) -> ImageFont.ImageFont:
    font_size = max(48, min(360, int(round(min(width, height) * 0.09))))
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _overlay_label(candidate_id: str) -> str:
    for prefix in ("cand-", "comp-"):
        if candidate_id.startswith(prefix):
            return candidate_id[len(prefix):]
    return candidate_id


def _flux_to_magnitude(flux: float, zero_point: float) -> float:
    if not math.isfinite(flux) or flux <= 0.0 or not math.isfinite(zero_point):
        return math.nan
    magnitude, _ = flux_to_mag(flux, 0.0)
    return float(magnitude) + zero_point


def _format_float(value: float, *, precision: int) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"
