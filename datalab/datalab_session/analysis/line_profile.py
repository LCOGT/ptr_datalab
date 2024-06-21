from skimage.measure import profile_line
from astropy.wcs import WCS

from datalab.datalab_session.util import scale_points
from datalab.datalab_session.util import get_hdu

# For creating an array of brightness along a user drawn line
def line_profile(input: dict):
  """
    Creates an array of luminosity values and the length of the line in arcseconds
  """
  sci_hdu = get_hdu(input['basename'], 'SCI')
  
  x_points, y_points = scale_points(input["height"], input["width"], sci_hdu.data.shape[0], sci_hdu.data.shape[1], x_points=[input["x1"], input["x2"]], y_points=[input["y1"], input["y2"]])
  
  # Line profile and distance in arcseconds
  line_profile = profile_line(sci_hdu.data, (x_points[0], y_points[0]), (x_points[1], y_points[1]), mode="constant", cval=-1)
  

  # Calculations for coordinates and angular distance
  try: 
    wcs = WCS(sci_hdu.header)

    start_sky_coord = wcs.pixel_to_world(x_points[0], y_points[0])
    end_sky_coord = wcs.pixel_to_world(x_points[1], y_points[1])

    arcsec_angle = start_sky_coord.separation(end_sky_coord).arcsecond

    start_coords = [start_sky_coord.ra.deg, start_sky_coord.dec.deg]
    end_coords = [end_sky_coord.ra.deg, end_sky_coord.dec.deg]

  except Exception as e:
    print(e)
    error = f'{input["basename"]} does not have a valid WCS header'
    # if theres no valid WCS solution then we default to using pixscale to calculate the angle, and no coordinates
    start_coords = None
    end_coords = None
    arcsec_angle = len(line_profile) * sci_hdu.header["PIXSCALE"]

  return {"line_profile": line_profile, "arcsec": arcsec_angle, "start_coords": start_coords, "end_coords": end_coords}
  