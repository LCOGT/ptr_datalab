"""Pure, framework-free centroiding algorithm.

An AstroImageJ-style (Howell marginal-sum) centroid with an iterative,
sigma-clipped background estimate. This module deliberately has no Django or I/O
dependencies so it can be reused by any caller (the interactive centroiding view
in centroiding.py and the aperture photometry pipeline both build on it).
"""

from dataclasses import dataclass
import logging
import math

import numpy as np

log = logging.getLogger()
log.setLevel(logging.INFO)


PIXELCENTER = 0.5


@dataclass(frozen=True)
class PlaneModel:
  c0: float
  c1: float
  c2: float

  def value_at(self, x: float, y: float) -> float:
    return self.c0 + self.c1 * x + self.c2 * y


@dataclass(frozen=True)
class BackgroundModel:
  mean: float
  peak: float
  plane: PlaneModel | None = None


@dataclass(frozen=True)
class CentroidResult:
  x: float
  y: float
  background: float
  peak: float
  success: bool = True
  message: str | None = None


def _pixel(image: np.ndarray, x: int, y: int) -> float:
  if y < 0 or y >= image.shape[0] or x < 0 or x >= image.shape[1]:
    return math.nan
  return float(image[y, x])

## Finds the maximum pixel value within a circular region around the center.
def _source_max(image: np.ndarray, x_center: float, y_center: float, radius: float) -> float:
  radius2 = radius * radius
  i1 = int(x_center - radius)
  i2 = int(x_center + radius)
  j1 = int(y_center - radius)
  j2 = int(y_center + radius)

  source_max = -math.inf
  for j in range(j1, j2 + 1):
    dj = j - y_center + PIXELCENTER
    for i in range(i1, i2 + 1):
      di = i - x_center + PIXELCENTER
      if di * di + dj * dj <= radius2:
        value = _pixel(image, i, j)
        if not math.isnan(value) and value > source_max:
          source_max = value
  return source_max


def _fit_plane(points: list[tuple[float, float, float]]) -> PlaneModel | None:
  if len(points) < 4:
    return None

  sum_1 = float(len(points))
  sum_x = sum(x for x, _, _ in points)
  sum_y = sum(y for _, y, _ in points)
  sum_xx = sum(x * x for x, _, _ in points)
  sum_yy = sum(y * y for _, y, _ in points)
  sum_xy = sum(x * y for x, y, _ in points)
  sum_z = sum(z for _, _, z in points)
  sum_xz = sum(x * z for x, _, z in points)
  sum_yz = sum(y * z for _, y, z in points)

  matrix = [
    [sum_1, sum_x, sum_y, sum_z],
    [sum_x, sum_xx, sum_xy, sum_xz],
    [sum_y, sum_xy, sum_yy, sum_yz],
  ]

 ## Solve the linear system using Gaussian elimination with partial pivoting.
  for pivot in range(3):
    pivot_row = max(range(pivot, 3), key=lambda row: abs(matrix[row][pivot]))
    if abs(matrix[pivot_row][pivot]) < 1e-12:
      return None
    if pivot_row != pivot:
      matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot]

    pivot_value = matrix[pivot][pivot]
    for col in range(pivot, 4):
      matrix[pivot][col] /= pivot_value

    for row in range(3):
      if row == pivot:
        continue
      factor = matrix[row][pivot]
      if factor == 0.0:
        continue
      for col in range(pivot, 4):
        matrix[row][col] -= factor * matrix[pivot][col]

  return PlaneModel(matrix[0][3], matrix[1][3], matrix[2][3])


def sigma_clipped_annulus_background(
  image: np.ndarray,
  x_center: float,
  y_center: float,
  r_inner: float,
  r_outer: float,
  *,
  remove_stars: bool = True,
  max_iterations: int = 9,
  tolerance: float = 0.1,
) -> tuple[float, list[tuple[float, float, float]]] | None:
  """Mean background of a circular annulus, with optional 2-sigma star removal.

  This is the single background estimator shared by the centroiding algorithm and
  the aperture photometry pipeline. It collects the annulus pixels, iteratively
  rejects bright outliers (stars) at 2 sigma, then averages the pixels that
  survive a final rejection pass against the converged mean/stdev.

  ``max_iterations``/``tolerance`` control the star-rejection convergence. The
  defaults (9, 0.1) match AstroImageJ's centroid background; the aperture
  photometry pipeline overrides them with AstroImageJ's photometer settings
  (100, 1e-4), which converge correctly for low background levels.

  Returns ``(mean, kept_pixels)`` where each kept pixel is a ``(di, dj, value)``
  tuple with ``di``/``dj`` the pixel-center offsets from ``(x_center, y_center)``
  (useful for plane fitting), or ``None`` if the annulus has no valid pixels.
  """
  r12 = r_inner * r_inner
  r22 = r_outer * r_outer
  i1 = int(x_center - r_outer)
  i2 = int(x_center + r_outer)
  j1 = int(y_center - r_outer)
  j2 = int(y_center + r_outer)

  annulus_pixels: list[tuple[float, float, float]] = []
  for j in range(j1, j2 + 1):
    dj = j - y_center + PIXELCENTER
    for i in range(i1, i2 + 1):
      di = i - x_center + PIXELCENTER
      radius2 = di * di + dj * dj
      if r12 <= radius2 <= r22:
        value = _pixel(image, i, j)
        if not math.isnan(value):
          annulus_pixels.append((di, dj, value))

  if not annulus_pixels:
    return None

  if remove_stars:
    back_mean = 0.0
    back2_mean = 0.0
    previous_back_mean = 0.0
    for iteration in range(max_iterations):
      back_stdev = math.sqrt(max(0.0, back2_mean - back_mean * back_mean))
      lower = back_mean - 2.0 * back_stdev
      upper = back_mean + 2.0 * back_stdev
      clipped = [
        value
        for _, _, value in annulus_pixels
        if iteration == 0 or (lower <= value <= upper)
      ]
      if clipped:
        back_mean = sum(clipped) / len(clipped)
        back2_mean = sum(value * value for value in clipped) / len(clipped)
      if abs(previous_back_mean - back_mean) < tolerance:
        break
      previous_back_mean = back_mean

    back_stdev = math.sqrt(max(0.0, back2_mean - back_mean * back_mean))
    lower = back_mean - 2.0 * back_stdev
    upper = back_mean + 2.0 * back_stdev
    kept = [
      (di, dj, value)
      for di, dj, value in annulus_pixels
      if lower <= value <= upper
    ]
  else:
    kept = list(annulus_pixels)

  background = sum(value for _, _, value in kept) / len(kept) if kept else 0.0
  return background, kept


def _background(
  image: np.ndarray,
  x_center: float,
  y_center: float,
  radius: float,
  r_back1: float,
  r_back2: float,
  remove_background_stars: bool,
  use_plane_background: bool,
) -> BackgroundModel:
  source_max = _source_max(image, x_center, y_center, radius)
  if r_back2 <= r_back1:
    return BackgroundModel(0.0, source_max)

  result = sigma_clipped_annulus_background(
    image,
    x_center,
    y_center,
    r_back1,
    r_back2,
    remove_stars=remove_background_stars,
  )
  if result is None:
    return BackgroundModel(0.0, source_max)

  background, kept = result
  plane = _fit_plane(kept) if use_plane_background else None
  return BackgroundModel(background, source_max - background, plane)


def _background_value(
  background_model: BackgroundModel,
  x_center: float,
  y_center: float,
  i: int,
  j: int,
) -> float:
  if background_model.plane is None:
    return background_model.mean
  return background_model.plane.value_at(
    i - x_center + PIXELCENTER,
    j - y_center + PIXELCENTER,
  )


def _failed_centroid(
  x: float,
  y: float,
  background: float,
  peak: float,
  message: str,
) -> CentroidResult:
  log.warning(f"Centroiding failed: {message}")
  return CentroidResult(x, y, background, peak, success=False, message=message)


def centroid(
  image: np.ndarray,
  x_click: float,
  y_click: float,
  radius: float,
  r_back1: float,
  r_back2: float,
  *,
  find_centroid: bool = True,
  remove_background_stars: bool = True,
  use_plane_background: bool = False,
) -> CentroidResult:
  image = np.asarray(image, dtype=float)
  x_center = x_click
  y_center = y_click
  radius = max(radius, 3.0)
  width = int(2.0 * radius)
  height = width

  i1 = int(x_click - radius)
  i2 = i1 + width
  j1 = int(y_click - radius)
  j2 = j1 + height

  x_start = x_center
  y_start = y_center
  background_model = _background(
    image,
    x_center,
    y_center,
    radius,
    r_back1,
    r_back2,
    remove_background_stars,
    use_plane_background,
  )

  still_moving = True
  iteration = 100 if find_centroid else 0
  while still_moving and iteration > 0:
    x_delta = 0.0
    y_delta = 0.0
    total_signal = 0.0
    samples = 0

    for j in range(j1, j2 + 1):
      for i in range(i1, i2 + 1):
        value = _pixel(image, i, j)
        if not math.isnan(value):
          total_signal += value - _background_value(background_model, x_center, y_center, i, j)
          samples += 1

    if samples == 0:
      return _failed_centroid(
        x_start,
        y_start,
        background_model.mean,
        background_model.peak,
        "No valid pixels in centroid box.",
      )

    i_bar = total_signal / (i2 - i1 + 1)
    j_bar = total_signal / (j2 - j1 + 1)

    weight_i = 0.0
    for i in range(i1, i2 + 1):
      column_signal = 0.0
      di = i - x_center + PIXELCENTER
      for j in range(j1, j2 + 1):
        value = _pixel(image, i, j)
        if not math.isnan(value):
          column_signal += value - _background_value(background_model, x_center, y_center, i, j)
      delta = column_signal - i_bar
      if delta > 0.0:
        weight_i += delta
        x_delta += delta * di

    weight_j = 0.0
    for j in range(j1, j2 + 1):
      row_signal = 0.0
      dj = j - y_center + PIXELCENTER
      for i in range(i1, i2 + 1):
        value = _pixel(image, i, j)
        if not math.isnan(value):
          row_signal += value - _background_value(background_model, x_center, y_center, i, j)
      delta = row_signal - j_bar
      if delta > 0.0:
        weight_j += delta
        y_delta += delta * dj

    if weight_i == 0.0 and weight_j == 0.0:
      return _failed_centroid(
        x_start,
        y_start,
        background_model.mean,
        background_model.peak,
        "Centroid calculation has zero weight in both dimensions.",
      )
    if weight_i == 0.0:
      return _failed_centroid(
        x_start,
        y_start,
        background_model.mean,
        background_model.peak,
        "Centroid calculation has zero weight in the x dimension.",
      )
    if weight_j == 0.0:
      return _failed_centroid(
        x_start,
        y_start,
        background_model.mean,
        background_model.peak,
        "Centroid calculation has zero weight in the y dimension.",
      )

    x_delta /= weight_i
    y_delta /= weight_j

    if find_centroid and (
      abs(x_center + x_delta - x_start) > width
      or abs(y_center + y_delta - y_start) > height
    ):
      return _failed_centroid(
        x_start,
        y_start,
        background_model.mean,
        background_model.peak,
        "Centroid repositioning exceeded centroid box size.",
      )

    if abs(x_delta) < 0.01 and abs(y_delta) < 0.01:
      still_moving = False

    if find_centroid:
      x_center += x_delta
      y_center += y_delta
      i1 = int(x_center) - width // 2
      i2 = i1 + width
      j1 = int(y_center) - height // 2
      j2 = j1 + height
      background_model = _background(
        image,
        x_center,
        y_center,
        radius,
        r_back1,
        r_back2,
        remove_background_stars,
        use_plane_background,
      )

    iteration -= 1

  return CentroidResult(
    x_center,
    y_center,
    background_model.mean,
    background_model.peak,
    message="Centroid calculation completed.",
  )
