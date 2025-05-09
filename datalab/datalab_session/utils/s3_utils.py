import logging
from pathlib import Path
import uuid
import requests
import os
import urllib.request
from contextlib import contextmanager

import boto3
from botocore.exceptions import ClientError

from django.conf import settings

from datalab.datalab_session.exceptions import ClientAlertException

log = logging.getLogger()
log.setLevel(logging.INFO)

def add_file_to_bucket(item_key: str, path: object) -> str:
  """
  Stores a fits into the operation bucket in S3

  Args:
    item_key -- name under which to store the fits file
    fits_buffer -- the fits file in a BytesIO buffer to add to the bucket

  Returns:
    A presigned url for the object just added to the bucket
  """
  s3 = boto3.client('s3')
  try:
    response = s3.upload_file(
      path,
      settings.DATALAB_OPERATION_BUCKET,
      item_key
    )
  except ClientError as e:
    log.error(f'Error uploading the operation output: {e}')
    raise ClientAlertException(f'Error uploading the operation output')

  return get_s3_url(item_key)

def get_s3_url(key: str, bucket: str = settings.DATALAB_OPERATION_BUCKET) -> str:
  """
  Gets a presigned url from the bucket using the key

  Args:
    item_key -- name to look up in the bucket

  Returns:
    A presigned url for the object or None
  """
  s3 = boto3.client('s3')

  try:
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucket,
            'Key': key
        },
        ExpiresIn = 60 * 60 * 24 * 30 # URL will be valid for 30 days
    )
  except ClientError as e:
    log.error(f'Could not generate url for {key}: {e}')
    raise ClientAlertException(f'Could not create url for {key}')

  return url

def key_exists(key: str) -> bool:
  """
  Checks if a given string exists as part of an object key in an S3 bucket.

  Args:
    bucket_name (str): The name of the S3 bucket.
    prefix (str): The string to look for in the object keys.

  Returns:
    bool: True if at least one object key contains the given prefix, False otherwise.
  """
  s3 = boto3.client('s3')
  response = s3.list_objects_v2(Bucket=settings.DATALAB_OPERATION_BUCKET, Prefix=key, MaxKeys=1)
  return 'Contents' in response

def get_archive_url(basename: str, archive: str = settings.ARCHIVE_API) -> dict:
  """
  Looks for the key as a prefix in the operations s3 bucket

  Args:
    basename -- name to query

  Returns:
    dict of archive fits urls
  """
  query_params = {'basename_exact': basename }

  headers = {
    'Authorization': f'Token {settings.ARCHIVE_API_TOKEN}'
  }

  response = requests.get(archive + '/frames/', params=query_params, headers=headers)

  try:
    response.raise_for_status()
    image_data = response.json()
    results = image_data.get('results', None)
  except requests.HTTPError as e:
    log.error(f"Error fetching data from the archive: {e}")
    raise ClientAlertException(f"Error fetching data from the archive")
  
  if not results:
    raise ClientAlertException(f"Could not find {basename} in the archive")

  fits_url = results[0].get('url', 'No URL found')
  return fits_url

@contextmanager
def get_fits(basename: str, source: str = 'archive'):
  """
  Returns a Fits File for the given basename from the source bucket
  """
  basename = basename.replace('-large', '').replace('-small', '')
  # Use a unique filename for each download to prevent collisions
  unique_name = f"{basename}_{uuid.uuid4().hex}.fits"
  fits_path = os.path.join(settings.TEMP_FITS_DIR, unique_name)

  # download the file if it isn't already downloaded in our temp directory
  if not os.path.isfile(fits_path):

    # create the tmp directory if it doesn't exist
    if not os.path.exists(settings.TEMP_FITS_DIR):
      os.makedirs(settings.TEMP_FITS_DIR, exist_ok=True)

    match source:
      case 'archive':
        fits_url = get_archive_url(basename)
      case 'datalab':
        s3_folder_path = f'{basename.split("-")[0]}/{basename}.fits'
        fits_url = get_s3_url(s3_folder_path)
      case _:
        raise ClientAlertException(f"Source {source} not recognized")

    urllib.request.urlretrieve(fits_url, fits_path)
  
  try:
    yield fits_path
  finally:
    Path(fits_path).unlink(missing_ok=True)

def save_files_to_s3(cache_key, format, file_paths: dict, index=None):
  """
  Save multiple files to S3, generating URLs and returning them in a structured output.
  **file_paths args should follow convention of <annotation>_<file_type>_path
  <annotation> will be appended to the bucket key for naming in S3
  path extension will determine file type
  """
  bucket_key = f'{cache_key}/{cache_key}-{index}' if index else f'{cache_key}/{cache_key}'
  output = {
    'basename': f'{cache_key}-{index}' if index else cache_key,
    'source': 'datalab',
    'type': format
  }
 
  for key, path in file_paths.items():
    parts = key.split('_')
    annotation = parts[0] if len(parts) == 3 else ''  # Extract annotation if available
    file_ext = path.split('.')[-1]  # Extract extension from filename

    output_key = f"{annotation}_url" if annotation else f"{file_ext}_url"
    s3_key = f"{bucket_key}-{annotation}.{file_ext}" if annotation else f"{bucket_key}.{file_ext}"

    log.info(f"Uploading {path} to {s3_key}")
    output[output_key] = add_file_to_bucket(s3_key, path)
    
  return output
