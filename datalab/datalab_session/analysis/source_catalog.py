from datalab.datalab_session.util import get_hdu, scale_points
import numpy as np



def source_catalog(input: dict):
  """
    Returns a dict representing the source catalog data with x,y coordinates and flux values
  """
  
  

  cat_hdu = get_hdu(input['basename'], 'CAT')
  sci_hdu = get_hdu(input['basename'], 'SCI')

  # The number of sources to send back to the frontend, default 20
  SOURCE_CATALOG_COUNT = min(20, len(cat_hdu.data["x"]))

  # get x,y and flux values for the first SOURCE_CATALOG_COUNT sources
  x_points = cat_hdu.data["x"][:SOURCE_CATALOG_COUNT]
  y_points = cat_hdu.data["y"][:SOURCE_CATALOG_COUNT]
  flux = cat_hdu.data["flux"][:SOURCE_CATALOG_COUNT]

  # scale the x_points and y_points from the fits pixel coords to the jpg coords
  fits_height, fits_width = np.shape(sci_hdu.data)
  x_points, y_points = scale_points(fits_height, fits_width, input['width'], input['height'], x_points=x_points, y_points=y_points)

  # we will be giving a list of dicts representing each source back to the frontend
  source_catalog_data = []
  for i in range(SOURCE_CATALOG_COUNT):
    source_catalog_data.append({
      "x": x_points[i],
      "y": y_points[i],
      "flux": flux[i].astype(int)
    })

  return source_catalog_data
