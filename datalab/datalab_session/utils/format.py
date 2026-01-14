class Format():
  """Class that provides format types used in datalab inputs and outputs"""

  FITS = 'fits' # A properly formatted FITS file with 2D image data and header ex. median output
  IMAGE = 'image' # Image only with no FITS file associated with it ex. color_image output
  STRING = 'string' # A single string of data.
  FLOAT = 'float' # Floating point number data.
  INT = 'int' # Integer number data.
  SOURCE = 'source' # A source in the night sky, with associated coordinates
  # TABLE = 'table' # A table of data ex. future astro_source output
  # JSON = 'json' # Raw JSON data
