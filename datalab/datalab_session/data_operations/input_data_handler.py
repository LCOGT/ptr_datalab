from contextlib import ExitStack

from astropy.io import fits

from datalab.datalab_session.utils.s3_utils import get_fits
from datalab.datalab_session.utils.file_utils import get_hdu

class InputDataHandler():
  """A class to read FITS files and provide access to the data.

  Attributes:
    basename (str): The basename of the FITS file.
    fits_file (str): The path to the FITS file.
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
    self.sci_data = get_hdu(self.fits_file, 'SCI').data
  
  def __del__(self):
    self.exit_stack.close()

  def __str__(self) -> str:
    with fits.open(self.fits_file) as hdul:
      return f"{self.basename}@{self.fits_file}\nHDU List\n{hdul.info()}"
  
  def get_hdu(self, extension: str=None):
    """Return an HDU from the FITS file.
    
    Args:
      extension (str): The extension to return from the FITS file. Default is 'SCI'.
    """
    return get_hdu(self.fits_file, extension)
