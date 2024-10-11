import tempfile
import numpy as np
from astropy.io import fits

from datalab.datalab_session.file_utils import create_jpgs
from datalab.datalab_session.s3_utils import save_fits_and_thumbnails


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
    
  def __init__(self, cache_key: str, data: np.array, comment: str=None) -> None:
      """Inits FITSOutputHandler with cache_key and data.
      
      Args:
        cache_key (str): The cache key for the FITS file, used as an ID when stored in S3.
        data (np.array): The data that will create the image HDU.
        comment (str): Optionally add a comment to add to the FITS file.
      """
      self.datalab_id = cache_key
      self.primary_hdu = fits.PrimaryHDU(header=fits.Header([('KEY', cache_key)]))
      self.image_hdu = fits.ImageHDU(data=data, name='SCI')
      if comment: self.set_comment(comment)

  def __str__(self) -> str:
    return f"Key: {self.datalab_id}\nData:\n{self.data}"

  def set_comment(self, comment: str):
    """Add a comment to the FITS file."""
    self.primary_hdu.header.add_comment(comment)
  
  def create_and_save_data_products(self, index: int=None, large_jpg_path: str=None, small_jpg_path: str=None):
    """Create the FITS file and save it to S3.

    This function can be called when you're done with the operation and would like to save the FITS file and jpgs in S3.
    It returns a datalab output dictionary that is formatted to be readable by the frontend.
    
    Args:
      index (int): Optionally add an index to the FITS file name. Appended to cache_key for multiple outputs.
      large_jpg (str): Optionally add a path to a large jpg to save, will not create a new jpg.
      small_jpg (str): Optionally add a path to a small jpg to save, will not create a new jpg.
    """
    hdu_list = fits.HDUList([self.primary_hdu, self.image_hdu])
    fits_output_path = tempfile.NamedTemporaryFile(suffix=f'{self.datalab_id}.fits').name
    hdu_list.writeto(fits_output_path, overwrite=True)

    # allow for operations to pregenerate the jpgs, ex. RGB stacking
    if not large_jpg_path or not small_jpg_path:
      large_jpg_path, small_jpg_path = create_jpgs(self.datalab_id, fits_output_path)

    return save_fits_and_thumbnails(self.datalab_id, fits_output_path, large_jpg_path, small_jpg_path, index)
