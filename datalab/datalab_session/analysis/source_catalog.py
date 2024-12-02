import numpy as np

from datalab.datalab_session.file_utils import get_hdu, scale_points
from datalab.datalab_session.s3_utils import get_fits

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
def source_catalog(input: dict):
  """
    Returns a dict representing the source catalog data with x,y coordinates and flux values
  """
  fits_path = get_fits(input['basename'], input['source'])

  cat_hdu = get_hdu(fits_path, 'CAT')
  sci_hdu = get_hdu(fits_path, 'SCI')

  # The number of sources to send back to the frontend, default 50
  SOURCE_CATALOG_COUNT = min(50, len(cat_hdu.data["x"]))

  # get x,y and flux values for the first SOURCE_CATALOG_COUNT sources
  x_points = cat_hdu.data["x"][:SOURCE_CATALOG_COUNT]
  y_points = cat_hdu.data["y"][:SOURCE_CATALOG_COUNT]
  flux = cat_hdu.data["flux"][:SOURCE_CATALOG_COUNT]

  # ra, dec values may or may not be present in the CAT hdu
  if "ra" in cat_hdu.data.names and "dec" in cat_hdu.data.names:
    ra = cat_hdu.data["ra"][:SOURCE_CATALOG_COUNT]
    dec = cat_hdu.data["dec"][:SOURCE_CATALOG_COUNT]
  else:
    # TODO: implement a fallback way to calculate ra, dec from x, y and WCS
    ra = np.zeros(SOURCE_CATALOG_COUNT)
    dec = np.zeros(SOURCE_CATALOG_COUNT)

  # scale the x_points and y_points from the fits pixel coords to the jpg coords
  fits_height, fits_width = np.shape(sci_hdu.data)
  x_points, y_points = scale_points(fits_height, fits_width, input['width'], input['height'], x_points=x_points, y_points=y_points)

  # create the list of source catalog objects
  source_catalog_data = []
  for i in range(SOURCE_CATALOG_COUNT):
    source_catalog_data.append({
      "x": x_points[i],
      "y": y_points[i],
      "flux": flux[i].astype(int),
      # Astronomical coordinates are formatted to 6 decimal places
      "ra": '%.6f'%(ra[i]),
      "dec": '%.6f'%(dec[i])
    })

  return source_catalog_data
