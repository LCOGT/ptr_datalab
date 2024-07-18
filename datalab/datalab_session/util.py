import requests
import logging
import os
import urllib.request

import boto3
from astropy.io import fits
import numpy as np

from django.conf import settings
from botocore.exceptions import ClientError

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
    raise ClientError(f'Error uploading the operation output')

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
    raise ClientError(f'Could not create url for {key}')

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
    raise requests.HTTPError(f"Error fetching data from the archive")
  
  if not results:
    raise FileNotFoundError(f"Could not find {basename} in the archive")

  fits_url = results[0].get('url', 'No URL found')
  return fits_url

def get_hdu(basename: str, extension: str = 'SCI', source: str = 'archive') -> list[fits.HDUList]:
  """
  Returns a HDU for the given basename from the source
  Will download the file to a tmp directory so future calls can open it directly
  Warning: this function returns an opened file that must be closed after use
  """

  # use the basename to fetch and create a list of hdu objects
  basename = basename.replace('-large', '').replace('-small', '')
  basename_file_path = os.path.join(settings.TEMP_FITS_DIR, basename)

  # download the file if it isn't already downloaded in our temp directory
  if not os.path.isfile(basename_file_path):

    # create the tmp directory if it doesn't exist
    if not os.path.exists(settings.TEMP_FITS_DIR):
      os.makedirs(settings.TEMP_FITS_DIR)

    match source:
      case 'archive':
        fits_url = get_archive_url(basename)
      case 'datalab':
        s3_folder_path = f'{basename.split("-")[0]}/{basename}.fits'
        fits_url = get_s3_url(s3_folder_path)
      case _:
        raise ValueError(f"Source {source} not recognized")

    urllib.request.urlretrieve(fits_url, basename_file_path)

  hdu = fits.open(basename_file_path)
  try:
    extension = hdu[extension]
  except KeyError:
    raise KeyError(f"{extension} Header not found in fits file {basename}")
  
  return extension

def create_fits(key: str, image_arr: np.ndarray) -> fits.HDUList:

  header = fits.Header([('KEY', key)])
  primary_hdu = fits.PrimaryHDU(header=header)
  image_hdu = fits.ImageHDU(data=image_arr, name='SCI')

  hdu_list = fits.HDUList([primary_hdu, image_hdu])

  return hdu_list

def stack_arrays(array_list: list):
  """
  Takes a list of numpy arrays, crops them to an equal shape, and stacks them to be a 3d numpy array

  """
  min_shape = min(arr.shape for arr in array_list)
  cropped_data_list = [arr[:min_shape[0], :min_shape[1]] for arr in array_list]

  stacked = np.stack(cropped_data_list, axis=2)

  return stacked

def scale_points(height_1: int, width_1: int, height_2: int, width_2: int, x_points=[], y_points=[], flip_y = False, flip_x = False):
  """
    Scales x_points and y_points from img_1 height and width to img_2 height and width
    Optionally flips the points on the x or y axis
  """
  if any([dim == 0 for dim in [height_1, width_1, height_2, width_2]]):
    raise ValueError("height and width must be non-zero")

  # normalize the points to be lists in case tuples or other are passed
  x_points = np.array(x_points)
  y_points = np.array(y_points)

  x_points = (x_points / width_1 * width_2).astype(int)
  y_points = (y_points / height_1 * height_2).astype(int)

  if flip_y:
    y_points = height_2 - y_points

  if flip_x:
    x_points = width_2 - x_points

  return x_points, y_points
