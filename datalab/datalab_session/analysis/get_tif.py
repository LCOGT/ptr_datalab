from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import create_tif, temp_file_manager
from datalab.datalab_session.utils.s3_utils import key_exists, add_file_to_bucket, get_s3_url
from datalab.datalab_session.utils.filecache import FileCache
from django.contrib.auth.models import User



def get_tif(input: dict, user: User):
  """
    Checks bucket for tif file and returns the url
    if the file doesn't exist, generates a new tif file
    input: dict
      basename: str
      source: str
  """

  basename = input["basename"]
  file_key = f'{basename}/{basename}.tif'

  # Check in bucket for tif file
  if(key_exists(file_key)):
    tif_url = get_s3_url(file_key)
  else:
    # If tif file doesn't exist, generate a new tif file
    try:
      file_path = FileCache().get_fits(basename, input.get('source', 'archive'), user)
    except TimeoutError as e:
      raise ClientAlertException(f"Download of {basename} timed out")

    with temp_file_manager(f'{basename}.tif') as tif_path:
      create_tif(file_path, tif_path)
      tif_url = add_file_to_bucket(file_key, tif_path)

  return {"tif_url": tif_url}
