import tempfile
import logging

from astropy.io import fits
import numpy as np
from fits2image.conversions import fits_to_jpg, fits_to_tif

from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.s3_utils import save_fits_and_thumbnails

log = logging.getLogger()
log.setLevel(logging.INFO)

def get_hdu(path: str, extension: str = 'SCI') -> list[fits.HDUList]:
  """
  Returns a HDU for the fits in the given path
  Warning: this function returns an opened file that must be closed after use
  """
  hdu = fits.open(path)
  try:
    extension = hdu[extension]
  except KeyError:
    raise ClientAlertException(f"{extension} Header not found in fits file at {path.split('/')[-1]}")
  
  return extension

def get_fits_dimensions(fits_file, extension: str = 'SCI') -> tuple:
  return fits.open(fits_file)[extension].shape

def create_fits(key: str, image_arr: np.ndarray) -> str:
  """
  Creates a fits file with the given key and image array
  Returns the the path to the fits_file
  """

  header = fits.Header([('KEY', key)])
  primary_hdu = fits.PrimaryHDU(header=header)
  image_hdu = fits.ImageHDU(data=image_arr, name='SCI')

  hdu_list = fits.HDUList([primary_hdu, image_hdu])
  fits_path = tempfile.NamedTemporaryFile(suffix=f'{key}.fits').name
  hdu_list.writeto(fits_path)

  return fits_path

def create_tif(key: str, fits_path: np.ndarray) -> str:
  """
    Creates a full sized TIFF file from a FITs
    Returns the path to the TIFF file
  """
  height, width = get_fits_dimensions(fits_path)
  tif_path = tempfile.NamedTemporaryFile(suffix=f'{key}.tif').name
  fits_to_tif(fits_path, tif_path, width=width, height=height)

  return tif_path

def create_jpgs(cache_key, fits_paths: str, color=False) -> list:
    """
    Create jpgs from fits files and save them to S3
    If using the color option fits_paths need to be in order R, G, B
    percent and cur_percent are used to update the progress of the operation
    """

    if not isinstance(fits_paths, list):
        fits_paths = [fits_paths]

    # create the jpgs from the fits files
    large_jpg_path      = tempfile.NamedTemporaryFile(suffix=f'{cache_key}-large.jpg').name
    thumbnail_jpg_path  = tempfile.NamedTemporaryFile(suffix=f'{cache_key}-small.jpg').name

    max_height, max_width = max(get_fits_dimensions(path) for path in fits_paths)

    fits_to_jpg(fits_paths, large_jpg_path, width=max_width, height=max_height, color=color)
    fits_to_jpg(fits_paths, thumbnail_jpg_path, color=color)

    return large_jpg_path, thumbnail_jpg_path

def crop_arrays(array_list: list):
  """
  Takes a list of numpy arrays from fits images and stacks them to be a 3d numpy array
  cropped since fits images can be different sizes
  """
  min_x = min(arr.shape[0] for arr in array_list)
  min_y = min(arr.shape[1] for arr in array_list)

  cropped_data_list = [arr[:min_x, :min_y] for arr in array_list]
  return cropped_data_list

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

def create_output(cache_key, np_array=None, fits_file=None, large_jpg=None, small_jpg=None, index=None):
  """
  A more automated way of creating output for a dev
  Dev can specify just a cache_key and np array and the function will create the fits and jpgs
  or the dev can pass the fits_file or jpgs and the function will save them
  """

  if np_array is not None and fits_file is None:
    fits_file = create_fits(cache_key, np_array)

  if not large_jpg or not small_jpg:
    large_jpg, small_jpg = create_jpgs(cache_key, fits_file)
  
  return save_fits_and_thumbnails(cache_key, fits_file, large_jpg, small_jpg, index)
