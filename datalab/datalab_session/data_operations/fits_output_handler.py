import tempfile
import os
import numpy as np
from astropy.io import fits

from datalab import settings
from datalab.datalab_session.utils.file_utils import create_jpgs, temp_file_manager
from datalab.datalab_session.utils.s3_utils import save_files_to_s3


class FITSOutputHandler():
  """A class to handle FITS output files and create jpgs.
  
  Class handles the creation of Datalab output for developers.
  The class inits with a cache_key and data, and creates a FITS file with the data.
  The FITS file is then saved to the cache and the large and small jpgs are created.

  Attributes:
    datalab_id (str): The cache key for the FITS file.
    primary_hdu (fits.PrimaryHDU): The primary HDU for the FITS file.
    image_hdu (fits.ImageHDU): The image HDU for the FITS file.
    data (np.array): The data for the image HDU.
  """
    
  def __init__(self, cache_key: str, data: np.array, comment: str=None, data_header: fits.Header=None) -> None:
      """Inits FITSOutputHandler with cache_key and data.
      
      Args:
        cache_key (str): The cache key for the FITS file, used as an ID when stored in S3.
        data (np.array): The data that will create the image HDU.
        comment (str): Optionally add a comment to add to the FITS file.
      """
      self.datalab_id = cache_key
      self.primary_hdu = fits.PrimaryHDU(header=fits.Header([('DLAB_KEY', cache_key)]))
      self.image_hdu = fits.ImageHDU(data=data, header=data_header, name='SCI')

      if comment: self.set_comment(comment)

  def __str__(self) -> str:
    return f"Key: {self.datalab_id}\nData:\n{self.data}"

  def set_comment(self, comment: str):
    """Add a comment to the FITS file."""
    self.primary_hdu.header.add_comment(comment)
  
  def create_and_save_data_products(self, format, index: int=None, large_jpg_path: str=None, small_jpg_path: str=None, tif_path: str=None):
    """
    When you're done with the operation and would like to save the FITS file and jpgs in S3. JPGs are required, any other file is optional.
    
    Args:
      index (int): Adds an index to the FITS file name for multiple outputs
      large_jpg (str): existing jpg, used in RGB stack
      small_jpg (str): existing jpg, used in RGB stack
      tif_path (str): optional tif file
    Returns:
      Datalab output dictionary that is formatted to be readable by the frontend
    """
    file_paths = {}
    hdu_list = fits.HDUList([self.primary_hdu, self.image_hdu])

    with tempfile.NamedTemporaryFile(suffix=f'{self.datalab_id}.fits', dir=settings.TEMP_FITS_DIR) as fits_output_file:
      # Create the output FITS file
      fits_output_path = fits_output_file.name
      hdu_list.writeto(fits_output_path, overwrite=True)

      # Create jpgs if not provided
      with temp_file_manager(f"{self.datalab_id}-large.jpg", f"{self.datalab_id}-small.jpg", dir=settings.TEMP_FITS_DIR) as (gen_large_jpg, gen_small_jpg):
        if not large_jpg_path or not small_jpg_path:
          create_jpgs(fits_output_path, gen_large_jpg, gen_small_jpg)

        if tif_path:
          file_paths['tif_path'] = tif_path
        
        file_paths['large_jpg_path'] = large_jpg_path or gen_large_jpg
        file_paths['small_jpg_path'] = small_jpg_path or gen_small_jpg
        file_paths['fits_path'] = fits_output_path

        return save_files_to_s3(self.datalab_id, format, file_paths, index)
