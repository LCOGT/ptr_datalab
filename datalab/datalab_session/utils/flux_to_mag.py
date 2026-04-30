import numpy as np


def flux_to_mag(flux, fluxerr):
  """
  Convert flux and fluxerr to magnitude and magnitude error.
  """
  conversion_factor = 2.5
  flux2mag = conversion_factor / np.log(10)

  flux_array = np.asarray(flux)
  fluxerr_array = np.asarray(fluxerr)

  if flux_array.ndim == 0:
    if flux_array <= 0:
      return None, None

    mag = -conversion_factor * np.log10(flux_array)
    magerr = flux2mag * (fluxerr_array / flux_array)
    return float(mag), float(magerr)

  mag = np.full(flux_array.shape, np.nan, dtype=float)
  magerr = np.full(flux_array.shape, np.nan, dtype=float)
  valid_flux = flux_array > 0

  mag[valid_flux] = -conversion_factor * np.log10(flux_array[valid_flux])
  magerr[valid_flux] = flux2mag * (fluxerr_array[valid_flux] / flux_array[valid_flux])

  return mag, magerr
