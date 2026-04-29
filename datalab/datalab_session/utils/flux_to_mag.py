import numpy as np


def flux_to_mag(flux, fluxerr):
  """
  Convert flux and fluxerr to magnitude and magnitude error.
  """
  conversion_factor = 2.5
  flux2mag = conversion_factor / np.log(10)

  if flux <= 0:
    return None, None

  mag = -conversion_factor * np.log10(flux)
  magerr = flux2mag * (fluxerr / flux)

  return mag, magerr
