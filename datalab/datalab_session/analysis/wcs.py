from astropy.wcs import WCS, WcsError
from django.contrib.auth.models import User

from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import get_hdu
from datalab.datalab_session.utils.filecache import FileCache


def wcs(input: dict, user: User):
  """
  Retrieves WCS information for a given FITS file.
  input = {
    basename (str): The name of the file to analyze
    source (str): Wether the file is in archive or datalab s3
  }
  """
  try:
    file_path = FileCache().get_fits(input['basename'], input['source'], user)
    sci_hdu = get_hdu(file_path, 'SCI')
    fits_dimensions = [sci_hdu.data.shape[0], sci_hdu.data.shape[1]]
  except TimeoutError as e:
    raise ClientAlertException(f"Download of {input['basename']} timed out")
  except TypeError as e:
    raise ClientAlertException(e)

  try:
    wcs = WCS(sci_hdu.header)
    wcs_solution = wcs.wcs
    wcs_cd = wcs_solution.cd
    output = {
        "crval": wcs_solution.crval.tolist(),
        "crpix": wcs_solution.crpix.tolist(),
        "cd1": [wcs_cd[0][0], wcs_cd[0][1]],
        "cd2": [wcs_cd[1][0], wcs_cd[1][1]],
        "fits_dimensions": fits_dimensions
    }
    return output
  except WcsError as e:
    raise ClientAlertException(e)
