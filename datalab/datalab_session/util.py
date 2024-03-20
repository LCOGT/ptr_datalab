import requests
import logging

import boto3

from django.conf import settings

log = logging.getLogger()
log.setLevel(logging.INFO)

def store_fits_output(item_key, fits_buffer):
  """
  Stores a fits into the operation bucket in S3

  Keyword Arguements:
  item_key -- name under which to store the fits file
  fits_buffer -- the fits file to add to the bucket
  """
  log.info(f'Adding {item_key} to {settings.DATALAB_OPERATION_BUCKET}')

  s3 = boto3.resource('s3')
  response = s3.Bucket(settings.DATALAB_OPERATION_BUCKET).put_object(Key = item_key, Body = fits_buffer.getvalue())
  return response

def get_archive_from_basename(basename: str) -> dict:
  """
  Querys and returns an archive file from the Archive

  Keyword Arguements:
  basename -- name to query
  """
  query_params = {'basename_exact': basename }

  response = requests.get(settings.ARCHIVE_API + '/frames/', params=query_params)

  try:
    image_data = response.json()
    results = image_data.get('results', None)
  except IndexError:
    log.error(f"No image found with specified basename: {basename}")
    raise FileNotFoundError

  return results
