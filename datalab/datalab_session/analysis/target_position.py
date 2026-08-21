from urllib.error import HTTPError

from django.contrib.auth.models import User

from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import get_fits_header
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.fits_metadata import target_radec_from_header


def target_position(input: dict, user: User):
  try:
    file_path = FileCache().get_fits(input['basename'], input.get('source', 'archive'), user)
    sci_header = get_fits_header(file_path, 'SCI')
  except TimeoutError:
    raise ClientAlertException(f"Download of {input['basename']} FITs timed out")
  except TypeError as e:
    raise ClientAlertException(e)
  except HTTPError:
    raise ClientAlertException(f"No FITs file found for this image")

  try:
    ra, dec = target_radec_from_header(sci_header)
  except ValueError as e:
    raise ClientAlertException(e)

  return {'ra': ra, 'dec': dec}
