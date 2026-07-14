import logging

import numpy as np
from django.core.cache import cache

from datalab.datalab_session.exceptions import ClientAlertException

log = logging.getLogger()
log.setLevel(logging.INFO)

GAIA_TABLE = 'gaiadr3.gaia_source'
GAIA_SYNTH_TABLE = 'gaiadr3.synthetic_photometry_gspc'
# Bailer-Jones et al. (2021) geometric distances (external.gaiaedr3_distance): a Bayesian
# posterior with a direction-dependent distance prior, so the estimate is always positive and
# well behaved even where the parallax is negative or low-significance.
# r_med_geo is the point estimate (parsecs); r_lo_geo / r_hi_geo are the 16th / 84th
# percentiles (asymmetric, 1-sigma-like bounds). source_ids match gaiadr3.gaia_source.
# Photogeometric columns (r_med_photogeo,...) are sharper where BP/RP photometry is good but bake in a Galactic CMD population model;
# geometric is the robust default and assumes nothing photometric about the sources.
GAIA_DISTANCE_TABLE = 'external.gaiaedr3_distance'
GAIA_DISTANCE_COLUMNS = ['r_med_geo', 'r_lo_geo', 'r_hi_geo']
GAIA_SOURCE_COLUMNS = ['ra', 'dec', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'parallax', 'parallax_error',
                       'phot_g_mean_mag']
# Synthetic SDSS-system AB magnitudes from the Gaia Synthetic Photometry Catalogue (computed
# from the BP/RP spectra, so only sources with published spectra have them, G <~ 17.65).
# GSPC standardizes griz to the SDSS system - within ~0.03 mag of the Pan-STARRS1 system the
# images are calibrated to, the same order as BANZAI's color-term-free zero-pointing
GAIA_SYNTH_BANDS = ['g_sdss', 'r_sdss', 'i_sdss', 'z_sdss']
# columns queried per source; the result dict also carries a {band}_mag_error for each synth band
GAIA_COLUMNS = GAIA_SOURCE_COLUMNS + GAIA_DISTANCE_COLUMNS + [f'{band}_mag' for band in GAIA_SYNTH_BANDS]
# systematic floor on synthetic magnitude errors: SDSS<->PS1 system difference + GSPC calibration
GAIA_SYNTH_MAG_ERROR_FLOOR = 0.03
# RUWE < 1.4 is the standard threshold for well behaved astrometric solutions from Lindegren's Gaia astrometry notes
GAIA_RUWE_LIMIT = 1.4
# Mean Gaia DR3 parallax zero-point offset (Lindegren et al. 2021): observed parallaxes
# run ~17 microarcsec too small, so add it back
GAIA_PARALLAX_ZERO_POINT_MAS = 0.017
# Gaia DR3 is a static catalog, so cached cone searches stay valid; cache them separately from
# the operation cache so unrelated input tweaks don't re-hit the TAP service
GAIA_CACHE_DURATION = 60 * 60 * 24 * 30  # 30 days

# membership_guess tuning
MIN_GAIA_MATCHES_FOR_GUESS = 10  # too few matched stars to call a clump
PM_HISTOGRAM_HALF_WIDTH = 25.0   # mas/yr searched around the median proper motion
PM_HISTOGRAM_BIN = 1.0           # mas/yr cell size, ~cluster clump scale
MIN_CLUMP_COUNT = 5              # peak cells below this are noise, not a cluster
PM_RADIUS_BOUNDS = (0.5, 5.0)    # mas/yr allowed range for the suggested clump radius


def gaia_cone_search(center_ra: float, center_dec: float, radius_arcmin: float) -> dict:
  """
  Gaia DR3 cone search around (center_ra, center_dec) degrees, returning a dict of numpy
  arrays for GAIA_COLUMNS with the RUWE cut and parallax zero-point correction applied.
  Missing values (e.g. no proper motion solution) are np.nan.

  Results are cached by field + radius. Failures raise ClientAlertException so the
  operation fails visibly and can be re-run, rather than caching a degraded result.
  """
  # versioned key: whenever GAIA_COLUMNS changes (v2 added phot_g_mean_mag, v3 added the
  # Bailer-Jones distance columns) a month-old cached response would be missing the new keys
  cache_key = f'gaia_dr3_cone_v3_{center_ra:.4f}_{center_dec:.4f}_{radius_arcmin:.3f}'
  gaia_data = cache.get(cache_key)
  if gaia_data is not None:
    return gaia_data

  gaia_data = _query_gaia(center_ra, center_dec, radius_arcmin)
  cache.set(cache_key, gaia_data, GAIA_CACHE_DURATION)
  return gaia_data


def _query_gaia(center_ra: float, center_dec: float, radius_arcmin: float) -> dict:
  # astroquery is imported lazily so that Django startup (which imports every data operation
  # to build available_operations) doesn't pay its import cost
  from astroquery.gaia import Gaia

  source_columns = ', '.join(f'g.{column}' for column in GAIA_SOURCE_COLUMNS)
  # raw Bailer-Jones column names
  distance_columns = ', '.join(f'd.{column}' for column in GAIA_DISTANCE_COLUMNS)
  synth_columns = ', '.join(f's.{band}_mag, s.{band}_flux, s.{band}_flux_error'
                            for band in GAIA_SYNTH_BANDS)
  query = f"""
    SELECT {source_columns}, {distance_columns}, {synth_columns}
    FROM {GAIA_TABLE} AS g
    LEFT JOIN {GAIA_DISTANCE_TABLE} AS d ON d.source_id = g.source_id
    LEFT JOIN {GAIA_SYNTH_TABLE} AS s ON s.source_id = g.source_id
    WHERE CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {center_ra}, {center_dec}, {radius_arcmin / 60.0})) = 1
    AND g.ruwe < {GAIA_RUWE_LIMIT}
  """
  try:
    # This launches off an async query to the GAIA service
    job = Gaia.launch_job_async(query, verbose=False)
    # This waits synchronously for the results of the GAIA query, which is fine since this operation happens in a worker.
    # If this takes longer than the timeout limit of the worker task (1 hour), this task will die.
    table = job.get_results()
  except Exception as e:
    log.error(f'Gaia cone search failed for ({center_ra}, {center_dec}) r={radius_arcmin}\': {e}')
    raise ClientAlertException('Could not fetch Gaia data for the cluster field - the Gaia archive may '
                               'be temporarily unavailable, try re-running the operation later.')

  def column_values(name):
    # Get rid of nans from the returned data
    return np.ma.filled(np.ma.masked_invalid(table[name].data), np.nan).astype(float)

  gaia_data = {column: column_values(column) for column in GAIA_COLUMNS}
  # Adjust the parallax by the adjustment factor
  gaia_data['parallax'] = gaia_data['parallax'] + GAIA_PARALLAX_ZERO_POINT_MAS
  # propagate synthetic flux errors to magnitude errors, on a systematic floor
  for band in GAIA_SYNTH_BANDS:
    flux = column_values(f'{band}_flux')
    flux_error = column_values(f'{band}_flux_error')
    with np.errstate(divide='ignore', invalid='ignore'):
      mag_error = 2.5 / np.log(10.0) * flux_error / flux
    gaia_data[f'{band}_mag_error'] = np.sqrt(mag_error ** 2 + GAIA_SYNTH_MAG_ERROR_FLOOR ** 2)

  log.info(f'Gaia cone search returned {len(gaia_data["ra"])} sources '
           f'for ({center_ra}, {center_dec}) r={radius_arcmin}\'')
  return gaia_data


def _member_distance_window(distance, distance_lo, distance_hi, finite_pm, members):
  """
  Builds a (distance_min, distance_max) parsec window from the clump members' Bailer-Jones
  distances, or (None, None) when distances weren't supplied or too few members have one.

  Members scatter around the true cluster distance from both real depth and measurement error,
  so the half-window is 3x the larger of the robust scatter of the r_med_geo point estimates and
  the members' typical per-star uncertainty (median (r_hi_geo - r_lo_geo) / 2). distance_min is
  clamped non-negative since Bailer-Jones distances are always positive.
  """
  if distance is None:
    return None, None
  distance = np.asarray(distance, dtype=float)
  distance_lo = np.asarray(distance_lo, dtype=float)
  distance_hi = np.asarray(distance_hi, dtype=float)

  member_distance = distance[finite_pm][members]
  member_lo = distance_lo[finite_pm][members]
  member_hi = distance_hi[finite_pm][members]
  has_distance = np.isfinite(member_distance) & (member_distance > 0)
  if has_distance.sum() < 3:
    return None, None

  distances = member_distance[has_distance]
  distance_median = np.median(distances)
  robust_scatter = 1.4826 * np.median(np.abs(distances - distance_median))
  # per-star measurement uncertainty floor, using the asymmetric lo/hi bounds where present
  half_widths = (member_hi[has_distance] - member_lo[has_distance]) / 2.0
  half_widths = half_widths[np.isfinite(half_widths) & (half_widths > 0)]
  per_star_sigma = np.median(half_widths) if len(half_widths) else 0.0
  distance_sigma = max(robust_scatter, per_star_sigma)

  distance_min = round(float(max(distance_median - 3.0 * distance_sigma, 0.0)), 1)
  distance_max = round(float(distance_median + 3.0 * distance_sigma), 1)
  return distance_min, distance_max


def estimate_membership(pmra, pmdec, parallax, distance=None, distance_lo=None, distance_hi=None):
  """
  Suggests a default cluster membership selection from the proper motions, parallaxes, and
  Bailer-Jones distances of the Gaia-matched stars: the dominant proper-motion clump (2D
  histogram peak, robust to the cluster being a minority of the field) plus parallax and
  distance windows around the clump members.

  distance / distance_lo / distance_hi are the Bailer-Jones r_med_geo point estimate and its
  r_lo_geo / r_hi_geo 16th / 84th percentile bounds (parsecs); pass them to also get a distance
  window. When omitted (or too few members have a distance) the distance window is left None.

  Returns {'pmra', 'pmdec', 'pm_radius', 'parallax_min', 'parallax_max', 'distance_min',
  'distance_max'} in mas(/yr) and parsecs, or None when there are too few stars or no clear
  clump - the user then selects manually. Negative parallaxes are tolerated throughout.
  """
  pmra = np.asarray(pmra, dtype=float)
  pmdec = np.asarray(pmdec, dtype=float)
  parallax = np.asarray(parallax, dtype=float)

  finite_pm = np.isfinite(pmra) & np.isfinite(pmdec)
  if finite_pm.sum() < MIN_GAIA_MATCHES_FOR_GUESS:
    return None
  pmra_finite = pmra[finite_pm]
  pmdec_finite = pmdec[finite_pm]

  # histogram the proper motions around their median and take the densest cell as the clump
  bins = np.arange(-PM_HISTOGRAM_HALF_WIDTH, PM_HISTOGRAM_HALF_WIDTH + PM_HISTOGRAM_BIN, PM_HISTOGRAM_BIN)
  median_pmra, median_pmdec = np.median(pmra_finite), np.median(pmdec_finite)
  histogram, pmra_edges, pmdec_edges = np.histogram2d(pmra_finite - median_pmra, pmdec_finite - median_pmdec, bins=(bins, bins))
  peak_index = np.unravel_index(np.argmax(histogram), histogram.shape)
  peak_pmra = median_pmra + (pmra_edges[peak_index[0]] + pmra_edges[peak_index[0] + 1]) / 2.0
  peak_pmdec = median_pmdec + (pmdec_edges[peak_index[1]] + pmdec_edges[peak_index[1] + 1]) / 2.0

  # judge the clump by the stars within a small neighborhood of the peak cell rather than the
  # single cell count, so a clump straddling bin edges isn't split below the threshold
  near_peak = np.hypot(pmra_finite - peak_pmra, pmdec_finite - peak_pmdec) <= 2.0 * PM_HISTOGRAM_BIN
  if near_peak.sum() < MIN_CLUMP_COUNT:
    return None

  # refine the clump center with the stars near the peak, then size the radius from their scatter
  clump_pmra = float(np.median(pmra_finite[near_peak]))
  clump_pmdec = float(np.median(pmdec_finite[near_peak]))
  clump_distances = np.hypot(pmra_finite[near_peak] - clump_pmra, pmdec_finite[near_peak] - clump_pmdec)
  robust_sigma = 1.4826 * np.median(np.abs(clump_distances - np.median(clump_distances)))
  pm_radius = float(np.clip(3.0 * max(robust_sigma, np.median(clump_distances)), *PM_RADIUS_BOUNDS))

  # parallax window from the clump members with measured parallaxes
  members = np.hypot(pmra_finite - clump_pmra, pmdec_finite - clump_pmdec) <= pm_radius
  member_parallaxes = parallax[finite_pm][members]
  member_parallaxes = member_parallaxes[np.isfinite(member_parallaxes)]
  if len(member_parallaxes) >= 3:
    parallax_median = np.median(member_parallaxes)
    parallax_sigma = max(1.4826 * np.median(np.abs(member_parallaxes - parallax_median)), 0.05)
    parallax_min = round(float(parallax_median - 3.0 * parallax_sigma), 4)
    parallax_max = round(float(parallax_median + 3.0 * parallax_sigma), 4)
  else:
    parallax_min = None
    parallax_max = None

  # distance window from the clump members' Bailer-Jones distances, when provided
  distance_min, distance_max = _member_distance_window(distance, distance_lo, distance_hi, finite_pm, members)

  return {
    'pmra': round(clump_pmra, 3),
    'pmdec': round(clump_pmdec, 3),
    'pm_radius': round(pm_radius, 3),
    'parallax_min': parallax_min,
    'parallax_max': parallax_max,
    'distance_min': distance_min,
    'distance_max': distance_max,
  }
