from datalab.datalab_session.util import create_tif, get_fits, add_file_to_bucket, key_exists, get_s3_url

def get_tif(input: dict):
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
    fits_path = get_fits(basename)
    tif_path = create_tif(basename, fits_path)
    tif_url = add_file_to_bucket(file_key, tif_path)

  return {"tif_url": tif_url}
