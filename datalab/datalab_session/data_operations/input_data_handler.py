from astropy.io import fits

from datalab.datalab_session.s3_utils import get_fits
from datalab.datalab_session.file_utils import get_hdu

class InputDataHandler():
  """A class to read FITS files and provide access to the data.

  The class inits with a basename and source, and reads the FITS file
  this data is then stored in the class attributes for easy access.

  Attributes:
    basename (str): The basename of the FITS file.
    fits_file (str): The path to the FITS file.
    sci_data (np.array): The data from the 'SCI' extension of the FITS file.
  """

  def __init__(self, basename: str, source: str = None) -> None:
    """Inits InputDataHandler with basename and source.
    
    Uses the basename to query the archive for the matching FITS file.
    Also can take a source argument to specify a different source for the FITS file.
    At the time of writing two common sources are 'datalab' and 'archive'.
    New sources will need to be added in the get_fits function in s3_utils.py.

    Args:
      basename (str): The basename of the FITS file.
      source (str): Optionally add a source to the FITS file in case it's not the LCO archive.
    """
    self.basename = basename
    self.fits_file = get_fits(basename, source)
    self.sci_data = get_hdu(self.fits_file, 'SCI').data

  def __str__(self) -> str:
    with fits.open(self.fits_file) as hdul:
      return f"{self.basename}@{self.fits_file}\nHDU List\n{self.hdul.info()}"
  
  def get_hdu(self, extension: str=None):
    """Return an HDU from the FITS file.
    
    Args:
      extension (str): The extension to return from the FITS file. Default is 'SCI'.
    """
    return get_hdu(self.fits_file, extension)
