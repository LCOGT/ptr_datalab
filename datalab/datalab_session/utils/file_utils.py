import tempfile
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from astropy.io import fits
import numpy as np
from fits2image.conversions import fits_to_jpg, fits_to_img

from datalab import settings
from datalab.datalab_session.exceptions import ClientAlertException

log = logging.getLogger()
log.setLevel(logging.INFO)

TIFF_EXTENSION = 'TIFF'

def get_hdu(path: str, extension: str = 'SCI', use_fsspec: bool = False) -> list[fits.HDUList]:
  """
  Returns a HDU for the fits in the given path
  Warning: this function returns an opened file that must be closed after use
  """
  with fits.open(path, use_fsspec=use_fsspec) as hdu:
    try:
      extension_copy = hdu[extension].copy()
    except KeyError:
      raise ClientAlertException(f"{extension} Header not found in fits file at {path.split('/')[-1]}")
    
    return extension_copy

def get_fits_dimensions(fits_file, extension: str = 'SCI') -> tuple:
  with fits.open(fits_file) as hdu:
    hdu_shape = hdu[extension].shape
    return hdu_shape

@contextmanager
def create_fits(key: str, image_arr: np.ndarray, comment=None) -> str:
  """
  Creates a fits file with the given key and image array
  Returns the the path to the fits_file
  """

  header = fits.Header([('KEY', key)])
  header.add_comment(comment) if comment else None
  primary_hdu = fits.PrimaryHDU(header=header)
  image_hdu = fits.CompImageHDU(data=image_arr, name='SCI')

  hdu_list = fits.HDUList([primary_hdu, image_hdu])
  fits_path = tempfile.NamedTemporaryFile(suffix=f'{key}.fits', dir=settings.TEMP_FITS_DIR).name
  hdu_list.writeto(fits_path, overwrite=True)

  try:
    yield fits_path
  finally:
    os.remove(fits_path)

@contextmanager
def temp_file_manager(*filenames, dir=None):
  """
  Context manager to handle multiple temporary files safely.
  
  Usage:
    with temp_file_manager("file1.tif", "file2.jpg") as paths:
      print(paths)  # List of full file paths
  """
  temp_dir = dir or tempfile.gettempdir()
  file_paths = [tempfile.NamedTemporaryFile(suffix=fname, dir=temp_dir, delete=False).name for fname in filenames]

  try:
    if len(file_paths) == 1:
      yield file_paths[0]
    else:
      yield file_paths
  finally:
    for path in file_paths:
      Path(path).unlink(missing_ok=True)

def create_jpgs(fits_paths: str, large_jpg_path, thumbnail_jpg_path, color=False, zmin=None, zmax=None) -> list:
  """
    Converts FITS images to JPEG images.
    If using the color option fits_paths need to be in order R, G, B
  """

  if not isinstance(fits_paths, list):
    fits_paths = [fits_paths]

  max_height, max_width = max(get_fits_dimensions(fp) for fp in fits_paths)

  fits_to_jpg(fits_paths, large_jpg_path, width=max_width, height=max_height, color=color, zmin=zmin, zmax=zmax)
  fits_to_jpg(fits_paths, thumbnail_jpg_path, color=color, zmin=zmin, zmax=zmax)

def create_tif(fits_paths: np.ndarray, tif_path, color=False, zmin=None, zmax=None) -> str:
  """
    Converts FITS images to a TIFF file.
    If using the color option fits_paths need to be in order R, G, B
  """
  if not isinstance(fits_paths, list):
    fits_paths = [fits_paths]

  max_height, max_width = max(get_fits_dimensions(fp) for fp in fits_paths)
  fits_to_img(fits_paths, tif_path, TIFF_EXTENSION, width=max_width, height=max_height, color=color, zmin=zmin, zmax=zmax)

def crop_arrays(array_list: list, flatten=False):
  """
    Takes a list of numpy arrays from fits images returns an array of them cropped to the max common size
    since fits images can be different sizes. If flatten is true, the arrays are flattened with ravel().
    Returns a tuple of the array of arrays with the common size (x,y)
  """
  min_x = min(arr.shape[0] for arr in array_list)
  min_y = min(arr.shape[1] for arr in array_list)

  if flatten:
      cropped_data_list = [arr[:min_x, :min_y].ravel() for arr in array_list]
  else:
    cropped_data_list = [arr[:min_x, :min_y] for arr in array_list]
  return cropped_data_list, (min_x, min_y)

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
