import requests
import logging

import boto3
from astropy.io import fits
import numpy as np

from django.conf import settings
from django.core.cache import cache

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
  log.info(f'Adding {item_key} to {settings.DATALAB_OPERATION_BUCKET}')

  s3 = boto3.client('s3')
  response = s3.upload_file(
    path,
    settings.DATALAB_OPERATION_BUCKET,
    item_key
  )

  return get_presigned_url(item_key)

def get_presigned_url(key: str) -> str:
  """
  Gets a presigned url from the operation bucket using the key

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
            'Bucket': settings.DATALAB_OPERATION_BUCKET,
            'Key': key
        },
        ExpiresIn = 60 * 60 * 24 * 30 # URL will be valid for 30 days
    )
  except:
    log.error(f'File {key} not found in bucket')
    return None

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

def get_archive_from_basename(basename: str) -> dict:
  """
  Looks for the key as a prefix in the operations s3 bucket

  Args:
    basename -- name to query

  Returns:
    dict of archive fits urls
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

def get_hdu(basename: str, extension: str = 'SCI') -> list[fits.HDUList]:
  """
  Returns a list of Sci HDUs for the given basenames
  Warning: this function returns an opened file that must be closed after use
  """

  # use the basename to fetch and create a list of hdu objects
  basename = basename.replace('-large', '').replace('-small', '')

  if cache.get(f'{basename}-{extension}') is not None:
    return cache.get(f'{basename}-{extension}')

  archive_record = get_archive_from_basename(basename)

  try:
    fits_url = archive_record[0].get('url', 'No URL found')
  except IndexError:
    RuntimeWarning(f"No image found with specified basename: {basename}")

  hdu = fits.open(fits_url, use_fsspec=True)
  cache.set(f'{basename}-{extension}', hdu[extension])
  return hdu[extension]

def create_fits(key: str, image_arr: np.ndarray) -> fits.HDUList:

  header = fits.Header([('KEY', key)])
  primary_hdu = fits.PrimaryHDU(header=header)
  image_hdu = fits.ImageHDU(image_arr)

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

def scale_points(small_img_width: int, small_img_height: int, img_array: list, points: list[tuple[int, int]]):
  """
    Scale the coordinates from a smaller image to the full sized fits so we know the positions of the coords on the 2dnumpy array
    Returns the list of tuple points with coords scaled for the numpy array
  """

  large_height, large_width = np.shape(img_array)

  # If the aspect ratios don't match we can't be certain where the point was
  if small_img_width / small_img_height != large_width / large_height:
    raise ValueError("Aspect ratios of the two images must match")

  width_scale = large_width / small_img_width
  height_scale = large_height / small_img_height

  points_array = np.array(points)
  scaled_points = np.int_(points_array * [width_scale, height_scale])

  return scaled_points
