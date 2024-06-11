from skimage.measure import profile_line

from datalab.datalab_session.util import scale_points
from datalab.datalab_session.util import get_hdu

# For creating an array of brightness along a user drawn line
def line_profile(input: dict):
  """
    Creates an array of luminosity values and the length of the line in arcseconds
  """
  sci_hdu = get_hdu(input['basename'], 'SCI')

  x_points, y_points = scale_points(input["height"], input["width"], sci_hdu.data.shape[0], sci_hdu.data.shape[1], x_points=[input["x1"], input["x2"]], y_points=[input["y1"], input["y2"]])
  line_profile = profile_line(sci_hdu.data, (x_points[0], y_points[0]), (x_points[1], y_points[1]), mode="constant", cval=-1)
  arcsec = len(line_profile) * sci_hdu.header["PIXSCALE"]

  return {"line_profile": line_profile, "arcsec": arcsec}
