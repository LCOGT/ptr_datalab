import numpy as np
from skimage.measure import profile_line

# For creating an array of brightness along a user drawn line
def line_profile(input, sci_hdu):
  """
    Creates an array of luminosity values and the length of the line in arcseconds
  """
  points = scale_points(input['width'], input['height'], sci_hdu.data, [(input["x1"], input["y1"]), (input["x2"], input["y2"])])
  line_profile = profile_line(sci_hdu.data, points[0], points[1], mode="constant", cval=-1)
  arcsec = len(line_profile) * sci_hdu.header["PIXSCALE"]

  return {"line_profile": line_profile, "arcsec": arcsec}


def scale_points(small_img_width, small_img_height, img_array, points: list[tuple[int, int]]):
  """
    Scale the coordinates from a smaller image to the full sized fits so we know the positions of the coords on the 2dnumpy array
    Returns the list of tuple points with coords scaled for the numpy array
  """

  large_height, large_width = np.shape(img_array)

  # If the aspect ratios don't match we can't be certain where the point was
  if small_img_width / small_img_height != large_width / large_height:
    raise ValueError("Aspect ratios of the two images must match")

  width_scale = large_width / small_img_width
  height_scale = large_height / small_img_height

  points_array = np.array(points)
  scaled_points = np.int_(points_array * [width_scale, height_scale])

  return scaled_points
