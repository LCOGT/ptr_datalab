import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import get_hdu, scale_points
from datalab.datalab_session.utils.filecache import FileCache

# Source catalog Function Definition
# ARGS: input (dict)
#   input = {
#     basename (str): The name of the file to analyze
#     height (int): The height of the image
#     width (int): The width of the image
#     source (str): The source of the file
#   }
# RETURNS: output (dict)
#   output = [{
#     x (int): The x coordinate of the source
#     y (int): The y coordinate of the source
#     flux (int): The flux value of the source
#     ra (float): The right ascension of the source
#     dec (float): The declination of the source
#   }]
# 
def source_catalog(input: dict, user: User):
  """
    Returns a dict representing the source catalog data with xwin,ywin coordinates and flux values
  """
  try:
    file_path = FileCache().get_fits(input['basename'], input.get('source', 'archive'), user)
  except TimeoutError as e:
    raise ClientAlertException(f"Download of {input['basename']} timed out")
  
  cat_hdu = get_hdu(file_path, 'CAT')
  sci_hdu = get_hdu(file_path, 'SCI')

  DECIMALS_OF_PRECISION = 6
  MAX_SOURCE_CATALOG_SIZE = min(len(cat_hdu.data["x"]), 1000)

  # get xwin,ywin and flux values
  # We get xwin and ywin because they provide more accurate centroid positions
  # Which in turn return more precise values for separation and positon angles for binary and blended stars
  x_points = cat_hdu.data["xwin"][:MAX_SOURCE_CATALOG_SIZE]
  y_points = cat_hdu.data["ywin"][:MAX_SOURCE_CATALOG_SIZE]
  x = cat_hdu.data["x"][:MAX_SOURCE_CATALOG_SIZE]
  y = cat_hdu.data["y"][:MAX_SOURCE_CATALOG_SIZE]
  flux = cat_hdu.data["flux"][:MAX_SOURCE_CATALOG_SIZE]
  # ra, dec values may or may not be present in the CAT hdu
  if "ra" in cat_hdu.data.names and "dec" in cat_hdu.data.names:
    ra = cat_hdu.data["ra"][:MAX_SOURCE_CATALOG_SIZE]
    dec = cat_hdu.data["dec"][:MAX_SOURCE_CATALOG_SIZE]
  else:
    # TODO: implement a fallback way to calculate ra, dec from x, y and WCS
    ra = None
    dec = None

  # scale the x_points and y_points from the fits pixel coords to the jpg coords
  fits_height, fits_width = np.shape(sci_hdu.data)
  x_points, y_points = scale_points(fits_height, fits_width, input['width'], input['height'], x_points=x_points, y_points=y_points)
  x, y = scale_points(fits_height, fits_width, input['width'], input['height'], x_points=x, y_points=y)

  # create the list of source catalog objects
  source_catalog_data = []
  for i in range(MAX_SOURCE_CATALOG_SIZE):
    source_data = {
      "x_win": x_points[i],
      "y_win": y_points[i],
      "x": x[i],
      "y": y[i],
      "flux": flux[i].astype(int)
    }
    if ra is not None and dec is not None:
      source_data["ra"] = f'%.{DECIMALS_OF_PRECISION}f' % (ra[i])
      source_data["dec"] = f'%.{DECIMALS_OF_PRECISION}f' % (dec[i])

    source_catalog_data.append(source_data)

  return source_catalog_data
