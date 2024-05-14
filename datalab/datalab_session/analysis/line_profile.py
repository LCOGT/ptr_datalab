from skimage.measure import profile_line

from datalab.datalab_session.util import scale_points

# For creating an array of brightness along a user drawn line
def line_profile(input: dict, sci_hdu: object):
  """
    Creates an array of luminosity values and the length of the line in arcseconds
  """
  points = scale_points(input['width'], input['height'], sci_hdu.data, [(input["x1"], input["y1"]), (input["x2"], input["y2"])])
  line_profile = profile_line(sci_hdu.data, points[0], points[1], mode="constant", cval=-1)
  arcsec = len(line_profile) * sci_hdu.header["PIXSCALE"]

  return {"line_profile": line_profile, "arcsec": arcsec}
