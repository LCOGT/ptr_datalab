from astropy.io import fits

from datalab.datalab_session.s3_utils import get_fits
from datalab.datalab_session.file_utils import get_hdu

class FITSFileReader:

  basename = None
  fits_file = None
  hdu_list = None

  def __init__(self, basename: str, source: str = None) -> None:
    self.basename = basename
    self.fits_file = get_fits(basename, source)
    self.hdu_list = fits.open(self.fits_file)

  def __str__(self) -> str:
    return f"{self.basename}@{self.fits_file}\nHDU List\n{self.hdu_list.info()}"
  
  @property
  def sci_data(self):
    return self.hdu_list['SCI'].data
  
  def hdu(self, extension: str):
    return get_hdu(self.fits_file, extension)
