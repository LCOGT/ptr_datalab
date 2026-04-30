import numpy as np

conversion_factor = 2.5
flux2mag = conversion_factor / np.log(10)


def _calculate_mag(flux, fluxerr):
  mag = -conversion_factor * np.log10(flux)
  magerr = flux2mag * (fluxerr / flux)

  return mag, magerr


def flux_to_mag_scalar(flux, fluxerr):
  """
  Convert scalar flux and fluxerr values to magnitude and magnitude error.
  """
  if flux <= 0:
    return None, None

  mag, magerr = _calculate_mag(flux, fluxerr)

  return float(mag), float(magerr)


def flux_to_mag_array(flux, fluxerr):
  """
  Convert flux and fluxerr arrays to magnitude and magnitude error arrays.
  """
  mag = np.full(flux.shape, np.nan, dtype=float)
  magerr = np.full(flux.shape, np.nan, dtype=float)
  valid_flux = flux > 0

  mag[valid_flux], magerr[valid_flux] = _calculate_mag(flux[valid_flux], fluxerr[valid_flux])

  return mag, magerr


def flux_to_mag(flux, fluxerr):
  """
  Convert flux and fluxerr to magnitude and magnitude error.
  """
  if np.isscalar(flux):
    return flux_to_mag_scalar(flux, fluxerr)

  return flux_to_mag_array(flux, fluxerr)
