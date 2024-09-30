import numpy as np

from datalab.datalab_session.file_utils import get_hdu, scale_points
from datalab.datalab_session.s3_utils import get_fits

def source_catalog(input: dict):
  """
    Returns a dict representing the source catalog data with x,y coordinates and flux values
  """
  fits_path = get_fits(input['basename'])

  cat_hdu = get_hdu(fits_path, 'CAT')
  sci_hdu = get_hdu(fits_path, 'SCI')

  # The number of sources to send back to the frontend, default 50
  SOURCE_CATALOG_COUNT = min(50, len(cat_hdu.data["x"]))

  # get x,y and flux values for the first SOURCE_CATALOG_COUNT sources
  x_points = cat_hdu.data["x"][:SOURCE_CATALOG_COUNT]
  y_points = cat_hdu.data["y"][:SOURCE_CATALOG_COUNT]
  flux = cat_hdu.data["flux"][:SOURCE_CATALOG_COUNT]
  ra = cat_hdu.data["ra"][:SOURCE_CATALOG_COUNT]
  dec = cat_hdu.data["dec"][:SOURCE_CATALOG_COUNT]

  # scale the x_points and y_points from the fits pixel coords to the jpg coords
  fits_height, fits_width = np.shape(sci_hdu.data)
  x_points, y_points = scale_points(fits_height, fits_width, input['width'], input['height'], x_points=x_points, y_points=y_points)

  # we will be giving a list of dicts representing each source back to the frontend
  source_catalog_data = []
  for i in range(SOURCE_CATALOG_COUNT):
    source_catalog_data.append({
      "x": x_points[i],
      "y": y_points[i],
      "flux": flux[i].astype(int),
      # truncate the ra and dec to 4 decimal places for readability
      "ra": '%.4f'%(ra[i]),
      "dec": '%.4f'%(dec[i])
    })

  return source_catalog_data
