import math

import numpy as np

from datalab.datalab_session.utils.centroiding import BackgroundModel

def measure_aperture(
    *,
    image: np.ndarray,
    x_center: float,
    y_center: float,
    aperture_radius_px: float,
    background_model: BackgroundModel,
    gain: float,
    read_noise: float,
    dark: float,
    error_class: type[Exception] = ValueError,
) -> dict[str, float]:
    """
        Measures source counts and uncertainty within a circular aperture, accounting for background
        subtraction and instrumental noise.

        Defines a circular aperture around the source and sums the flux within it, weighting each
        pixel by its fractional overlap with the aperture. Subtracts the background contribution
        (the constant background_model.mean) to obtain the net source counts, and derives the source
        uncertainty from the source and background levels and the instrumental parameters (gain,
        read noise, dark current). A fitted plane on the background model, if present, is not applied
        here -- photometry always uses the constant mean background.

        Returns the net source counts, source uncertainty, mean background per pixel, peak pixel
        value, and effective number of source and background pixels.
    """
    if not math.isfinite(gain) or gain <= 0.0:
        raise error_class("Detector gain must be positive and finite for aperture photometry.")
    height, width = image.shape
    source_radius = aperture_radius_px
    bck_cnt = float(max(int(background_model.effective_pixels), 1))
    if background_model.effective_pixels <= 0.0:
        raise error_class("Background annulus does not contain any valid pixels.")

    mean_background_per_pixel = max(background_model.mean, 0.0)
    source_sum = 0.0
    source_area = 0.0
    peak_pixel_value = -math.inf

    source_min_x = max(int(math.floor(x_center - source_radius - 1)), 0)
    source_max_x = min(int(math.ceil(x_center + source_radius + 1)), width - 1)
    source_min_y = max(int(math.floor(y_center - source_radius - 1)), 0)
    source_max_y = min(int(math.ceil(y_center + source_radius + 1)), height - 1)
    for j in range(source_min_y, source_max_y + 1):
        for i in range(source_min_x, source_max_x + 1):
            value = float(image[j, i])
            if not math.isfinite(value):
                continue
            fraction = fractional_pixel_overlap(i, j, x_center, y_center, source_radius)
            if fraction <= 0.0:
                continue
            peak_pixel_value = max(peak_pixel_value, value)
            source_sum += value * fraction
            source_area += fraction

    if source_area <= 0.0:
        raise error_class("Source aperture does not contain any valid pixels.")

    background_total = mean_background_per_pixel * source_area
    net_source = source_sum - background_total
    src = max(net_source, 0.0)
    bck = mean_background_per_pixel
    s_cnt = max(source_area, 0.0)
    src_cnt = source_area
    source_uncertainty = math.sqrt(
        (src * gain)
        + s_cnt * (1.0 + src_cnt / bck_cnt) * (bck * gain + dark + read_noise * read_noise + gain * gain * 0.083521)
    ) / gain

    return {
        "net_source_counts": net_source,
        "source_uncertainty": source_uncertainty,
        "mean_background_per_pixel": mean_background_per_pixel,
        "peak_pixel_value": peak_pixel_value,
        "effective_source_pixels": source_area,
        "effective_background_pixels": bck_cnt,
    }


def fractional_pixel_overlap(
    i: int,
    j: int,
    x_center: float,
    y_center: float,
    radius: float,
    substeps: int = 5,
) -> float:
    """
        Approximates the fractional overlap of a pixel at (i, j) with a circular aperture centered at (x_center, y_center) with the given radius.
    """
    inside = 0
    total = substeps * substeps
    for sy in range(substeps):
        y = j + (sy + 0.5) / substeps
        for sx in range(substeps):
            x = i + (sx + 0.5) / substeps
            dx = x - x_center
            dy = y - y_center
            if dx * dx + dy * dy <= radius * radius:
                inside += 1
    return inside / total
