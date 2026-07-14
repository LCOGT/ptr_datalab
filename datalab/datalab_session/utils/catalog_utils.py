import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import get_hdu

log = logging.getLogger()
log.setLevel(logging.INFO)


def extract_calibrated_catalog(fits_path: str, basename: str = '') -> dict:
  """
  Reads the CAT HDU of a fits file and returns full-precision numpy arrays of the
  source positions and zero-point calibrated photometry: {ra, dec, mag, magerr}

  - Requires calibrated mag/magerr columns, there is no instrumental flux fallback
    since an un-zero-pointed magnitude makes a color meaningless
  - Computes ra/dec from the x/y columns and the SCI WCS when the CAT lacks them
  - Drops rows with non-finite values in any returned column
  """
  cat_data = get_hdu(fits_path, 'CAT').data

  if 'mag' not in cat_data.names or 'magerr' not in cat_data.names:
    raise ClientAlertException(f'{basename} has no zero-point calibrated magnitudes (CAT mag/magerr), '
                               'which the HR Diagram requires. Try an image in a calibrated filter.')

  mag = np.asarray(cat_data['mag'], dtype=float)
  magerr = np.asarray(cat_data['magerr'], dtype=float)

  if 'ra' in cat_data.names and 'dec' in cat_data.names:
    ra = np.asarray(cat_data['ra'], dtype=float)
    dec = np.asarray(cat_data['dec'], dtype=float)
  else:
    ra, dec = _ra_dec_from_wcs(fits_path, cat_data, basename)

  valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag) & np.isfinite(magerr)
  return {'ra': ra[valid], 'dec': dec[valid], 'mag': mag[valid], 'magerr': magerr[valid]}


def _ra_dec_from_wcs(fits_path: str, cat_data, basename: str = ''):
  """
  Fallback for CAT HDUs without ra/dec columns: compute them from the catalog x/y
  pixel positions and the SCI extension's WCS solution
  """
  if 'x' not in cat_data.names or 'y' not in cat_data.names:
    raise ClientAlertException(f'{basename} catalog has no ra/dec or x/y columns to locate its sources')

  sci_header = get_hdu(fits_path, 'SCI').header
  wcs = WCS(sci_header)
  if not wcs.has_celestial:
    raise ClientAlertException(f'{basename} catalog has no ra/dec columns and the image has no WCS solution to compute them')

  # origin=1: source extractor catalog x/y positions use the FITS 1-based convention
  ra, dec = wcs.all_pix2world(cat_data['x'], cat_data['y'], 1)
  return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


def cross_match_one_to_one(catalog_a: dict, catalog_b: dict, match_radius_arcsec: float):
  """
  Nearest-neighbor sky match of catalog_a sources to catalog_b sources within match_radius_arcsec,
  deduplicated to one-to-one pairs (when multiple a sources share the same nearest b source,
  only the closest pair is kept, avoiding double counted stars in crowded fields).

  Catalogs are dicts holding equal-length 'ra' and 'dec' arrays in degrees.
  Returns (a_indices, b_indices) index arrays of the matched pairs.
  """
  if len(catalog_a['ra']) == 0 or len(catalog_b['ra']) == 0:
    return np.array([], dtype=int), np.array([], dtype=int)

  coords_a = SkyCoord(ra=catalog_a['ra'] * u.deg, dec=catalog_a['dec'] * u.deg)
  coords_b = SkyCoord(ra=catalog_b['ra'] * u.deg, dec=catalog_b['dec'] * u.deg)
  nearest_b, sep2d, _ = coords_a.match_to_catalog_sky(coords_b)
  sep_arcsec = np.atleast_1d(sep2d.arcsec)
  nearest_b = np.atleast_1d(nearest_b)

  # match_to_catalog_sky returns a nearest neighbor unconditionally, so apply the separation cut,
  # then keep only the closest a for each b to enforce one-to-one matching
  best_a_for_b = {}
  for a_index in np.flatnonzero(sep_arcsec <= match_radius_arcsec):
    b_index = int(nearest_b[a_index])
    if b_index not in best_a_for_b or sep_arcsec[a_index] < sep_arcsec[best_a_for_b[b_index]]:
      best_a_for_b[b_index] = int(a_index)

  pairs = sorted((a_index, b_index) for b_index, a_index in best_a_for_b.items())
  a_indices = np.array([pair[0] for pair in pairs], dtype=int)
  b_indices = np.array([pair[1] for pair in pairs], dtype=int)
  return a_indices, b_indices


def cone_filter(ra, dec, center_ra: float, center_dec: float, radius_arcmin: float):
  """
  Returns a boolean mask of the (ra, dec) points within radius_arcmin of the center point.
  All coordinates in degrees.
  """
  center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
  points = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg)
  return center.separation(points).arcmin <= radius_arcmin


def wcs_contains_point(fits_path: str, ra: float, dec: float) -> bool:
  """
  Best-effort check of whether (ra, dec) falls inside the image's WCS footprint.
  Returns True when the WCS is missing or unusable so an inconclusive check never blocks an operation.
  """
  try:
    sci_header = get_hdu(fits_path, 'SCI').header
    wcs = WCS(sci_header)
    if not wcs.has_celestial:
      return True
    return bool(wcs.footprint_contains(SkyCoord(ra=ra * u.deg, dec=dec * u.deg)))
  except Exception as e:
    log.warning(f'Could not check WCS footprint for {fits_path}: {e}')
    return True
