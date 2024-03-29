import requests
import logging
from io import BytesIO

import boto3
from PIL import Image
import numpy as np
from fits2image.conversions import fits_to_jpg

from django.conf import settings

log = logging.getLogger()
log.setLevel(logging.INFO)

def add_file_to_bucket(item_key: str, file: object, file_format:str) -> object:
  """
  Stores a fits into the operation bucket in S3

  Keyword Arguements:
  item_key -- name under which to store the fits file
  fits_buffer -- the fits file in a BytesIO buffer to add to the bucket
  """
  log.info(f'Adding {item_key} to {settings.DATALAB_OPERATION_BUCKET}')

  buffer = BytesIO()

  if file_format == 'JPEG':
    file.convert('RGB').save(buffer, format=file_format)
  elif file_format == 'FITS':
    file.writeto(buffer)
  else:
    log.error(f'Unknown file format {format}')
    raise ValueError(f'add_file_to_bucket cant process {format} files')

  buffer.seek(0)

  s3 = boto3.client('s3')
  response = s3.upload_fileobj(
    buffer,
    settings.DATALAB_OPERATION_BUCKET,
    item_key
  )
  return response

def get_archive_from_basename(basename: str) -> dict:
  """
  Queries and returns an archive file from the Archive

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

def numpy_to_thumbnails(img_arr: np.ndarray) -> dict:
  """
  Transforms a 2D numpy array into full res and thumbnail jpg image

  Keyword Arguements:
  basename -- name to query
  """
  THUMBNAIL_HEIGHT = 256
  THUMBNAIL_WIDTH = 256
  try:
    log.info(f'img_arr dtype: {img_arr.dtype}')
    log.info(f'img_arr shape: {img_arr.shape}')
    full_size_img = Image.fromarray(img_arr, mode='F')
    log.info(full_size_img.mode)
    full_size_img.show()
    thumbnail_img = full_size_img.copy()
    thumbnail_img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
    return {'full': full_size_img, 'thumbnail': thumbnail_img}
  except:
    log.error(f'Failed to convert array {img_arr.shape} to jpgs')
    raise OSError
