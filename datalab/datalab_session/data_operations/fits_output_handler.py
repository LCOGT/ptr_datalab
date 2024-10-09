import tempfile
import numpy as np
from astropy.io import fits

from datalab.datalab_session.file_utils import create_jpgs
from datalab.datalab_session.s3_utils import save_fits_and_thumbnails


class FITSOutputHandler():
    
  def __init__(self, key: str, data: np.array, comment: str=None) -> None:
      self.key = key
      self.primary_hdu = fits.PrimaryHDU(header=fits.Header([('KEY', key)]))
      self.image_hdu = fits.ImageHDU(data=data, name='SCI')
      if comment: self.set_comment(comment)

  def __str__(self) -> str:
    return f"Key: {self.key}\nData:\n{self.data}"

  def set_comment(self, comment: str):
    self.primary_hdu.header.add_comment(comment)

  def set_sci_data(self, new_data: np.array):
    self.image_hdu.data = new_data
  
  def create_save_fits(self, index: int=None, large_jpg: str=None, small_jpg: str=None):
    hdu_list = fits.HDUList([self.primary_hdu, self.image_hdu])
    fits_output_path = tempfile.NamedTemporaryFile(suffix=f'{self.key}.fits').name
    hdu_list.writeto(fits_output_path, overwrite=True)

    # allow for operations to pregenerate the jpgs, ex. RGB stacking
    if not large_jpg or not small_jpg:
      large_jpg, small_jpg = create_jpgs(self.key, fits_output_path)

    return save_fits_and_thumbnails(self.key, fits_output_path, large_jpg, small_jpg, index)
