from dataclasses import dataclass
import logging
import math

import numpy as np


log = logging.getLogger()
log.setLevel(logging.INFO)

HALF_PIXEL = 0.5


@dataclass(frozen=True)
class PlaneModel:
  """
    Linear background plane model of the form z = c0 + c1 * x + c2 * y.
    Fitted from annulus pixels.
  """

  c0: float
  c1: float
  c2: float

  def value_at(self, x: float, y: float) -> float:
    """
      Returns the value of the plane at the given (x, y) coordinates.
    """
    return self.c0 + self.c1 * x + self.c2 * y


@dataclass(frozen=True)
class BackgroundModel:
  """ 
    Background estimate for an aperture photometry measurement. 
    Describes only the background. Source brightness values are calculated separately.
  """
  mean: float
  plane: PlaneModel | None = None
  effective_pixels: float = 0.0


@dataclass(frozen=True)
class CentroidResult:
  """
    Result of centroiding a source.

    x: centroid x coordinate in FITS pixel coords
    y: centroid y coordinate in FITS pixel coords
    peak: estimated peak value of the source (background subtracted)
    background_model: background model used for the centroiding and photometry
      (the background level is background_model.mean, not duplicated here)
    success: whether the centroiding was successful
    message: optional message describing the result or any issues encountered
  """
  x: float
  y: float
  peak: float
  background_model: BackgroundModel | None = None
  success: bool = True
  message: str | None = None

def _pixel(image: np.ndarray, x: int, y: int) -> float:
  """
    Returns the pixel value at the given coordinates, or NaN if the coordinates are out of bounds.
  """
  if y < 0 or y >= image.shape[0] or x < 0 or x >= image.shape[1]:
    return math.nan
  return float(image[y, x])


def _aperture_peak(image: np.ndarray, x_center: float, y_center: float, radius: float) -> float:
  """
    Returns the brightest valid pixel value within a circular aperture.
  """
  radius2 = radius * radius
  i1 = int(x_center - radius)
  i2 = int(x_center + radius)
  j1 = int(y_center - radius)
  j2 = int(y_center + radius)

  peak = -math.inf
  for j in range(j1, j2 + 1):
    dj = j - y_center + HALF_PIXEL
    for i in range(i1, i2 + 1):
      di = i - x_center + HALF_PIXEL
      if di * di + dj * dj <= radius2:
        value = _pixel(image, i, j)
        if not math.isnan(value) and value > peak:
          peak = value
  return peak

def _fit_plane(points: list[tuple[float, float, float]]) -> PlaneModel | None:
  """
    Fits a linear background plane to sampled points in the form (x, y, z) using least squares.
    Returns a PlaneModel if successful, or None if the fit fails (e.g., not enough points or rank deficiency).
  """
  if len(points) < 4:
    return None

  pts = np.asarray(points, dtype=float)
  x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
  design = np.column_stack((np.ones_like(x), x, y))
  coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
  if rank < 3:
    return None
  return PlaneModel(*coefficients)


def calculate_background_model(
  image: np.ndarray,
  x_center: float,
  y_center: float,
  radius: float,
  r_back1: float,
  r_back2: float,
  remove_background_stars: bool,
  use_plane_background: bool,
  max_iterations: int = 9,
  tolerance: float = 0.1,
) -> BackgroundModel:
  """
    Calculates the local background from the annulus around the source.
    Returns a BackgroundModel containing the mean background value, an optional fitted plane, and the number of effective pixels used in the calculation.

    max_iterations/tolerance control the 2-sigma star-rejection convergence. The defaults (9, 0.1
    counts) are AstroImageJ's centroid-background settings. AstroImageJ's photometer uses tighter
    settings (100, 1e-4) for low background levels; those are exposed here so they can be switched
    on for near-zero-background inputs (e.g. background-subtracted or difference images) without
    re-plumbing.
  """
  if r_back2 <= r_back1:
    return BackgroundModel(mean=0.0)

  r12 = r_back1 * r_back1
  r22 = r_back2 * r_back2
  i1 = int(x_center - r_back2)
  i2 = int(x_center + r_back2)
  j1 = int(y_center - r_back2)
  j2 = int(y_center + r_back2)

  annulus_pixels: list[tuple[float, float, float]] = []
  for j in range(j1, j2 + 1):
    dj = j - y_center + HALF_PIXEL
    for i in range(i1, i2 + 1):
      di = i - x_center + HALF_PIXEL
      radius2 = di * di + dj * dj
      if r12 <= radius2 <= r22:
        value = _pixel(image, i, j)
        if not math.isnan(value):
          annulus_pixels.append((di, dj, value))

  back_mean = 0.0
  back2_mean = 0.0
  if remove_background_stars:
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
    if not remove_background_stars or (lower <= value <= upper)
  ]

  background = sum(value for _, _, value in kept) / len(kept) if kept else 0.0
  plane = _fit_plane(kept) if use_plane_background else None
  return BackgroundModel(
    mean=background,
    plane=plane,
    effective_pixels=float(len(kept)),
  )


def _background_value(
  background_model: BackgroundModel,
  x_center: float,
  y_center: float,
  i: int,
  j: int,
) -> float:
  """
    Returns the background value for a pixel using either the mean or fitted plane from the background model
  """
  if background_model.plane is None:
    return background_model.mean
  return background_model.plane.value_at(
    i - x_center + HALF_PIXEL,
    j - y_center + HALF_PIXEL,
  )


def _failed_centroid(
  x: float,
  y: float,
  peak: float,
  background_model: BackgroundModel,
  message: str,
) -> CentroidResult:
  """
    Builds a failed centroid result while preserving the original click position and diagnostics.
  """
  log.warning(f"Centroiding failed: {message}")
  return CentroidResult(
    x,
    y,
    peak,
    background_model=background_model,
    success=False,
    message=message,
  )


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
  """
    Finds a centroid (AIJ-style) around an initial source position (x_click, y_click).

    Input coords are in FITS pixel coordinates.

    Returns a CentroidResult containing the centroid position, localbackground estimate, peak value, and any relevant messages.
  """
  # No dtype here: forcing float64 would copy the entire frame on every call, and this runs once
  # per comparison candidate per frame. Pixels are read as Python floats, so any numeric dtype works.
  image = np.asarray(image)
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
  background_model = calculate_background_model(
    image,
    x_center,
    y_center,
    radius,
    r_back1,
    r_back2,
    remove_background_stars,
    use_plane_background,
  )
  raw_peak = _aperture_peak(image, x_center, y_center, radius)
  peak = raw_peak - background_model.mean

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
        peak,
        background_model,
        "No valid pixels in centroid box.",
      )

    i_bar = total_signal / (i2 - i1 + 1)
    j_bar = total_signal / (j2 - j1 + 1)

    weight_i = 0.0
    for i in range(i1, i2 + 1):
      column_signal = 0.0
      di = i - x_center + HALF_PIXEL
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
      dj = j - y_center + HALF_PIXEL
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
        peak,
        background_model,
        "Centroid calculation has zero weight in both dimensions.",
      )
    if weight_i == 0.0:
      return _failed_centroid(
        x_start,
        y_start,
        peak,
        background_model,
        "Centroid calculation has zero weight in the x dimension.",
      )
    if weight_j == 0.0:
      return _failed_centroid(
        x_start,
        y_start,
        peak,
        background_model,
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
        peak,
        background_model,
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
      background_model = calculate_background_model(
        image,
        x_center,
        y_center,
        radius,
        r_back1,
        r_back2,
        remove_background_stars,
        use_plane_background,
      )
      raw_peak = _aperture_peak(image, x_center, y_center, radius)
      peak = raw_peak - background_model.mean

    iteration -= 1

  return CentroidResult(
    x_center,
    y_center,
    peak,
    background_model=background_model,
    message="Centroid calculation completed.",
  )
