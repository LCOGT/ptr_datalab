from skimage.measure import profile_line

from datalab.datalab_session.util import scale_flip_points

# For creating an array of brightness along a user drawn line
def line_profile(input: dict, sci_hdu: object):
  """
    Creates an array of luminosity values and the length of the line in arcseconds
  """
  points = scale_flip_points(input['width'], input['height'], sci_hdu.data, [(input["x1"], input["y1"]), (input["x2"], input["y2"])])
  line_profile = profile_line(sci_hdu.data, points[0], points[1], mode="constant", cval=-1)
  arcsec = len(line_profile) * sci_hdu.header["PIXSCALE"]

  return {"line_profile": line_profile, "arcsec": arcsec}

def debug_point_on_sci_data(x, y, sci_hdu: object):
  """
    Debugging function to check a point (x,y) on the sci data has the same value as the point cross checked in DS9
  """
  print(f"data: {sci_hdu.data[y, x]}")
