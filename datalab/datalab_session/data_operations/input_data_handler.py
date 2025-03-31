from contextlib import ExitStack
import gc

from astropy.io import fits

from datalab.datalab_session.utils.s3_utils import get_fits
from datalab.datalab_session.utils.file_utils import get_hdu

class InputDataHandler():
  """A class to read FITS files and provide access to the data.

  Attributes:
    basename (str): The basename of the FITS file.
    fits_file (str): The path to the FITS file.
    sci_hdu (fits.HDU): The HDU from the 'SCI' extension of the FITS file.
    sci_data (np.array): The data from the 'SCI' extension of the FITS file.
  """

  def __init__(self, basename: str, source: str = None) -> None:
    """
    Supported sources are 'datalab' and 'archive'
    New sources will need to be added in get_fits

    Args:
      basename (str): The basename query for the FITS file
      source (str): location of the fits file
    """
    self.basename = basename
    self.source = source
    self.exit_stack = ExitStack()
    self.fits_file = self.exit_stack.enter_context(get_fits(basename, source))
    self.sci_hdu = get_hdu(self.fits_file, 'SCI')
    self.sci_data = self.sci_hdu.data
  
  def __del__(self):
    self.exit_stack.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # Using this as a context manager will ensure memory is returned when we are done with the file
    del self.sci_hdu
    del self.sci_data
    self.exit_stack.close()
    del self.fits_file
    gc.collect()

  def __str__(self) -> str:
    with fits.open(self.fits_file) as hdul:
      return f"{self.basename}@{self.fits_file}\nHDU List\n{hdul.info()}"

  def get_hdu(self, extension: str=None):
    """Return an HDU from the FITS file.
    
    Args:
      extension (str): The extension to return from the FITS file. Default is 'SCI'.
    """
    return get_hdu(self.fits_file, extension)
