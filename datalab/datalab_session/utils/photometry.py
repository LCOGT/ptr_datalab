import math
from typing import Any

import numpy as np

## Measure_aperture does the following:
# 1. It defines a circular annulus around the source position and collects the pixel values
#    within the annulus to estimate the background level. It performs iterative sigma clipping
#    to exclude outliers from the background estimation.
# 2. It defines a circular aperture around the source position and calculates the total flux
#    within the aperture, accounting for fractional pixel overlap. It also identifies the peak
#    pixel value within the aperture.
# 3. It computes the net source counts by subtracting the estimated background contribution from
#    the total flux in the aperture. It also calculates the uncertainty in the source measurement
#    based on the source counts, background level, and instrumental parameters (gain, read noise, dark current).
def measure_aperture(
    *,
    image: np.ndarray,
    x_center: float,
    y_center: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    gain: float,
    read_noise: float,
    dark: float,
    error_class: type[Exception] = ValueError,
) -> dict[str, float]:
    height, width = image.shape
    source_radius = aperture_radius_px
    annulus_inner_r2 = annulus_inner_radius_px * annulus_inner_radius_px
    annulus_outer_r2 = annulus_outer_radius_px * annulus_outer_radius_px

    min_x = max(int(math.floor(x_center - annulus_outer_radius_px - 1)), 0)
    max_x = min(int(math.ceil(x_center + annulus_outer_radius_px + 1)), width - 1)
    min_y = max(int(math.floor(y_center - annulus_outer_radius_px - 1)), 0)
    max_y = min(int(math.ceil(y_center + annulus_outer_radius_px + 1)), height - 1)

    annulus_pixels: list[float] = []
    for j in range(min_y, max_y + 1):
        dy = j - y_center + 0.5
        for i in range(min_x, max_x + 1):
            dx = i - x_center + 0.5
            r2 = dx * dx + dy * dy
            if annulus_inner_r2 <= r2 <= annulus_outer_r2:
                value = float(image[j, i])
                if math.isfinite(value):
                    annulus_pixels.append(value)

    if not annulus_pixels:
        raise error_class("Background annulus does not contain any valid pixels.")

    clipped = np.asarray(annulus_pixels, dtype=float)
    back_mean = 0.0
    back2_mean = 0.0
    previous_back_mean = 0.0
    for iteration in range(9):
        back_stdev = math.sqrt(max(0.0, back2_mean - back_mean * back_mean))
        lower = back_mean - 2.0 * back_stdev
        upper = back_mean + 2.0 * back_stdev
        used = clipped if iteration == 0 else clipped[(clipped >= lower) & (clipped <= upper)]
        if used.size:
            back_mean = float(np.mean(used))
            back2_mean = float(np.mean(used * used))
        if abs(previous_back_mean - back_mean) < 0.1:
            clipped = used
            break
        previous_back_mean = back_mean
        clipped = used

    mean_background_per_pixel = max(back_mean, 0.0)
    peak = -math.inf
    source_sum = 0.0
    source_area = 0.0

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
            source_sum += value * fraction
            source_area += fraction
            if value > peak:
                peak = value

    if source_area <= 0.0:
        raise error_class("Source aperture does not contain any valid pixels.")

    background_total = mean_background_per_pixel * source_area
    net_source = source_sum - background_total
    src = max(net_source, 0.0)
    bck = mean_background_per_pixel
    s_cnt = max(source_area, 0.0)
    src_cnt = source_area if clipped.size > 0 else 0.0
    bck_cnt = float(max(int(clipped.size), 1))
    source_uncertainty = math.sqrt(
        (src * gain)
        + s_cnt * (1.0 + src_cnt / bck_cnt) * (bck * gain + dark + read_noise * read_noise + gain * gain * 0.083521)
    ) / gain

    return {
        "net_source_counts": net_source,
        "source_uncertainty": source_uncertainty,
        "mean_background_per_pixel": mean_background_per_pixel,
        "peak_pixel_value": peak,
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
