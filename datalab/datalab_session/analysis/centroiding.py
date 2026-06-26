import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy.wcs import WCS, WcsError

from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import get_hdu, scale_points
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.analysis.centroiding_core import (
    BackgroundModel,
    CentroidResult,
    PIXELCENTER,
    PlaneModel,
    centroid,
)

if TYPE_CHECKING:
  from django.contrib.auth.models import User

log = logging.getLogger()
log.setLevel(logging.INFO)


def centroiding(input: dict, user: 'User'):
  """
    Finds an AIJ-like Howell centroid for a clicked source position.
    input = {
      basename (str): The name of the file to analyze
      height (int): The displayed image height
      width (int): The displayed image width
      x (float): Click x coordinate in displayed image space
      y (float): Click y coordinate in displayed image space
      radius (float): Centroid radius
      r_back1 (float): Inner background annulus radius
      r_back2 (float): Outer background annulus radius
    }
  """
  try:
    file_path = FileCache().get_fits(input['basename'], input.get('source', 'archive'), user)
    sci_hdu = get_hdu(file_path, 'SCI')
  except TimeoutError:
    raise ClientAlertException(f"Download of {input['basename']} timed out")
  except TypeError as e:
    raise ClientAlertException(f'Error: {e}')

  image = np.asarray(sci_hdu.data, dtype=float)
  if image.ndim != 2:
    message = f"Centroiding requires a 2D image, received shape {image.shape}."
    log.error(message)
    raise ClientAlertException(message)

  fits_height, fits_width = image.shape
  x_points, y_points = scale_points(
    input['height'],
    input['width'],
    fits_height,
    fits_width,
    x_points=[input['x']],
    y_points=[input['y']],
  )

  result = centroid(
    image,
    x_click=float(x_points[0]),
    y_click=float(y_points[0]),
    radius=float(input.get('radius', 8.0)),
    r_back1=float(input.get('r_back1', 10.0)),
    r_back2=float(input.get('r_back2', 15.0)),
    find_centroid=bool(input.get('find_centroid', True)),
    remove_background_stars=bool(input.get('remove_background_stars', True)),
    use_plane_background=bool(input.get('use_plane_background', False)),
  )

  output_x, output_y = scale_points(
    fits_height,
    fits_width,
    input['height'],
    input['width'],
    x_points=[result.x],
    y_points=[result.y],
  )

  ra = None
  dec = None
  try:
    wcs = WCS(sci_hdu.header)
    if wcs.get_axis_types()[0].get('coordinate_type') is None:
      raise WcsError("No valid WCS solution")
    sky_coord = wcs.pixel_to_world(result.y - 1, result.x - 1)
    ra = float(sky_coord.ra.deg)
    dec = float(sky_coord.dec.deg)
  except (AttributeError, IndexError, KeyError, TypeError, ValueError, WcsError):
    log.info(f"No valid WCS solution for centroiding on {input['basename']}")
    pass

  return {
    'x': float(output_x[0]),
    'y': float(output_y[0]),
    'ra': ra,
    'dec': dec,
    'background': result.background,
    'peak': result.peak,
    'success': result.success,
    'message': result.message,
  }
