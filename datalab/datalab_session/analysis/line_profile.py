from skimage.measure import profile_line
from astropy.wcs import WCS
from astropy.wcs import WcsError
from astropy import coordinates

from datalab.datalab_session.file_utils import scale_points, get_hdu

# For creating an array of brightness along a user drawn line
def line_profile(input: dict):
  """
    Creates an array of luminosity values and the length of the line in arcseconds
    input = {
      basename (str): The name of the file to analyze
      height (int): The height of the image
      width (int): The width of the image
      x1 (int): The x coordinate of the starting point
      y1 (int): The y coordinate of the starting point
      x2 (int): The x coordinate of the ending point
      y2 (int): The y coordinate of the ending point
    }
  """
  sci_hdu = get_hdu(input['basename'], 'SCI')

  x_points, y_points = scale_points(input["height"], input["width"], sci_hdu.data.shape[0], sci_hdu.data.shape[1], x_points=[input["x1"], input["x2"]], y_points=[input["y1"], input["y2"]])

  # Line profile and distance in arcseconds
  line_profile = profile_line(sci_hdu.data, (x_points[0], y_points[0]), (x_points[1], y_points[1]), mode="constant", cval=-1)


  # Calculations for coordinates and angular distance
  try:
    wcs = WCS(sci_hdu.header)

    if(wcs.get_axis_types()[0].get('coordinate_type') == None):
      raise WcsError("No valid WCS solution")

    start_sky_coord = wcs.pixel_to_world(x_points[0], y_points[0])
    end_sky_coord = wcs.pixel_to_world(x_points[1], y_points[1])

    arcsec_angle = start_sky_coord.separation(end_sky_coord).arcsecond

    start_coords = [start_sky_coord.ra.deg, start_sky_coord.dec.deg]
    end_coords = [end_sky_coord.ra.deg, end_sky_coord.dec.deg]
    position_angle = coordinates.position_angle(start_sky_coord.ra, start_sky_coord.dec,
                                                end_sky_coord.ra, end_sky_coord.dec).deg
  except WcsError:
    # no valid WCS solution
    start_coords = None
    end_coords = None
    position_angle = None

    try:
      # attempt using pixscale to calculate the angle
      arcsec_angle = len(line_profile) * sci_hdu.header["PIXSCALE"]
    except KeyError as e:
      # no valid WCS solution, and no pixscale
      arcsec_angle = None

  return {"line_profile": line_profile, "arcsec": arcsec_angle, "start_coords": start_coords, "end_coords": end_coords, "position_angle": position_angle}
