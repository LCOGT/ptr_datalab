import logging

import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.catalog_utils import (
  cone_filter,
  cross_match_one_to_one,
  extract_calibrated_catalog,
  wcs_contains_point,
)
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.gaia import estimate_membership, gaia_cone_search

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class HRDiagram(BaseDataOperation):
  # Fixed blue<->red match tolerance. Deliberately not a wizard input: every input is hashed
  # into the cache key, so exposing it would fragment the cache and force full recomputes
  MATCH_RADIUS_ARCSEC = 2.0
  # Tighter tolerance for matching our stars to Gaia. Gaia DR3 positions are epoch 2016.0,
  # so 1 arcsec also tolerates ~a decade of <100 mas/yr proper motion
  GAIA_MATCH_RADIUS_ARCSEC = 1.0
  # Keep only the brightest N matched stars to bound the output payload
  MAX_OUTPUT_SOURCES = 5000
  # Also emit up to this many of the brightest Gaia sources that did NOT match an image star
  # (flagged gaia_only): the cluster's proper-motion clump and parallax peak stand out far
  # better against the full Gaia field than against just the image-matched subset
  MAX_GAIA_ONLY_SOURCES = 5000
  DEFAULT_SEARCH_RADIUS_ARCMIN = 15.0
  # Disjoint blue/red filter options so the wizard can only produce valid, wavelength-ordered
  # (color = blue - red) pairs. BANZAI only reduces gp/rp/ip/zs, so gp is the only blue choice
  BLUE_FILTER_OPTIONS = ['gp']
  RED_FILTER_OPTIONS = ['rp', 'ip', 'zs']
  # Gaia-only stars get CMD photometry from Gaia's synthetic SDSS-system magnitudes (see
  # utils/gaia.py for the ~0.03 mag SDSS<->PS1 caveat baked into their errors)
  GAIA_SYNTH_MAG = {'gp': 'g_sdss', 'rp': 'r_sdss', 'ip': 'i_sdss', 'zs': 'z_sdss'}
  PROGRESS_STEPS = {
    'BLUE_CATALOG_DONE': 0.2,
    'RED_CATALOG_DONE': 0.4,
    'CROSS_MATCH_DONE': 0.5,
    'GAIA_QUERY_DONE': 0.80,
    'GAIA_MATCH_DONE': 0.95,
    'OUTPUT_DONE': 1.0,
  }

  @staticmethod
  def name():
    return 'HR Diagram'

  @staticmethod
  def description():
    return """Builds a color-magnitude diagram (an observational HR diagram) of a star cluster from two images of it taken in different filters.

The calibrated photometry catalogs of the two images are cross-matched by sky position, giving each star a color (blue - red magnitude) and a brightness. The output CMD can then be analyzed to estimate the cluster's distance, reddening, age, and metallicity by fitting isochrones."""

  @staticmethod
  def wizard_description():
    return {
      'name': HRDiagram.name(),
      'description': HRDiagram.description(),
      'category': 'image',
      'inputs': {
        'blue_filter_files': {
          'name': 'Blue Band Image',
          'description': 'The bluer filter image of the cluster',
          'type': Format.FITS,
          'single_filter': True,
          'filter_options': HRDiagram.BLUE_FILTER_OPTIONS,
          'minimum': 1,
          'maximum': 1,
        },
        'red_filter_files': {
          'name': 'Red Band Image',
          'description': 'The redder filter image of the cluster (color = blue - red)',
          'type': Format.FITS,
          'single_filter': True,
          'filter_options': HRDiagram.RED_FILTER_OPTIONS,
          'minimum': 1,
          'maximum': 1,
        },
        'cluster': {
          'name': 'Cluster Center',
          'description': 'The center of the star cluster to analyze',
          'type': Format.SOURCE,
        },
        'search_radius_arcmin': {
          'name': 'Search Radius (arcmin)',
          'description': 'Include stars within this radius of the cluster center',
          'type': Format.FLOAT,
          'default': HRDiagram.DEFAULT_SEARCH_RADIUS_ARCMIN,
        },
      },
    }

  def _validate_cluster(self):
    cluster = self.input_data.get('cluster') or {}
    try:
      cluster_ra = float(cluster.get('ra'))
      cluster_dec = float(cluster.get('dec'))
    except (TypeError, ValueError):
      raise ClientAlertException(f'Operation {self.name()} requires a cluster center with ra and dec.')
    return cluster.get('name', ''), cluster_ra, cluster_dec

  def _validate_search_radius(self):
    try:
      search_radius_arcmin = float(self.input_data.get('search_radius_arcmin') or HRDiagram.DEFAULT_SEARCH_RADIUS_ARCMIN)
    except (TypeError, ValueError):
      raise ClientAlertException('The search radius must be a number of arcminutes.')
    if search_radius_arcmin <= 0:
      raise ClientAlertException('The search radius must be greater than zero arcminutes.')
    return search_radius_arcmin

  def _band_catalog(self, image: dict, user: User) -> dict:
    """ Downloads one band's fits file and extracts its calibrated source catalog """
    basename = image.get('basename')
    try:
      fits_path = FileCache().get_fits(basename, image.get('source', 'archive'), user)
    except TimeoutError:
      raise ClientAlertException(f'Download of {basename} timed out')

    catalog = extract_calibrated_catalog(fits_path, basename=basename)
    if len(catalog['ra']) == 0:
      raise ClientAlertException(f'No calibrated sources found in the catalog of {basename}.')
    catalog['fits_path'] = fits_path
    return catalog

  def operate(self, submitter: User):
    blue_image = self._validate_inputs(input_key='blue_filter_files')[0]
    red_image = self._validate_inputs(input_key='red_filter_files')[0]
    blue_filter = blue_image.get('filter', blue_image.get('primary_optical_element', '')) or ''
    red_filter = red_image.get('filter', red_image.get('primary_optical_element', '')) or ''
    cluster_name, cluster_ra, cluster_dec = self._validate_cluster()
    search_radius_arcmin = self._validate_search_radius()

    log.info(f'HR Diagram for cluster ({cluster_ra}, {cluster_dec}) from images '
             f'{blue_image["basename"]} ({blue_filter}) and {red_image["basename"]} ({red_filter})')

    blue_catalog = self._band_catalog(blue_image, submitter)
    self.set_operation_progress(HRDiagram.PROGRESS_STEPS['BLUE_CATALOG_DONE'])
    red_catalog = self._band_catalog(red_image, submitter)
    self.set_operation_progress(HRDiagram.PROGRESS_STEPS['RED_CATALOG_DONE'])

    # cheap pre-flight to catch mistyped coordinates before cross-matching
    if not wcs_contains_point(blue_catalog['fits_path'], cluster_ra, cluster_dec) \
        and not wcs_contains_point(red_catalog['fits_path'], cluster_ra, cluster_dec):
      raise ClientAlertException('The cluster center is outside both image footprints - '
                                 'check that the cluster coordinates match your images.')

    blue_indices, red_indices = cross_match_one_to_one(blue_catalog, red_catalog, HRDiagram.MATCH_RADIUS_ARCSEC)
    if len(blue_indices) == 0:
      raise ClientAlertException('No stars matched between the two bands - '
                                 'check that both images cover the same field.')
    self.set_operation_progress(HRDiagram.PROGRESS_STEPS['CROSS_MATCH_DONE'])

    # the red (y-axis) band positions are the canonical star positions
    ra = red_catalog['ra'][red_indices]
    dec = red_catalog['dec'][red_indices]
    blue_mag = blue_catalog['mag'][blue_indices]
    blue_magerr = blue_catalog['magerr'][blue_indices]
    red_mag = red_catalog['mag'][red_indices]
    red_magerr = red_catalog['magerr'][red_indices]

    in_cone = cone_filter(ra, dec, cluster_ra, cluster_dec, search_radius_arcmin)
    if not in_cone.any():
      raise ClientAlertException(f'No matched stars within {search_radius_arcmin} arcminutes of the '
                                 'cluster center - try a larger search radius.')
    n_stars_matched = int(in_cone.sum())

    # keep the brightest stars first when capping the output size
    brightest = np.argsort(red_mag[in_cone])[:HRDiagram.MAX_OUTPUT_SOURCES]
    ra, dec = ra[in_cone][brightest], dec[in_cone][brightest]
    blue_mag, blue_magerr = blue_mag[in_cone][brightest], blue_magerr[in_cone][brightest]
    red_mag, red_magerr = red_mag[in_cone][brightest], red_magerr[in_cone][brightest]

    # Gaia enrichment: proper motions + parallaxes for cluster membership selection.
    # A Gaia failure raises and fails the operation (re-runnable) rather than silently
    # caching a membership-less result for 30 days
    gaia_data = gaia_cone_search(cluster_ra, cluster_dec, search_radius_arcmin)
    self.set_operation_progress(HRDiagram.PROGRESS_STEPS['GAIA_QUERY_DONE'])
    star_indices, gaia_indices = cross_match_one_to_one({'ra': ra, 'dec': dec}, gaia_data,
                                                        HRDiagram.GAIA_MATCH_RADIUS_ARCSEC)
    gaia_for_star = {int(star): int(gaia) for star, gaia in zip(star_indices, gaia_indices)}

    # the brightest Gaia sources without an image counterpart still map the cluster's
    # kinematics, so they are emitted too (flagged gaia_only) for the membership plots
    matched_gaia = np.asarray(gaia_indices, dtype=int)
    unmatched_gaia = np.setdiff1d(np.arange(len(gaia_data['ra'])), matched_gaia)
    unmatched_g_mag = gaia_data['phot_g_mean_mag'][unmatched_gaia]
    by_brightness = np.argsort(np.where(np.isfinite(unmatched_g_mag), unmatched_g_mag, np.inf))
    gaia_only_indices = unmatched_gaia[by_brightness][:HRDiagram.MAX_GAIA_ONLY_SOURCES]

    # the membership guess uses the same star set the membership plots will show
    membership_pool = np.concatenate([matched_gaia, gaia_only_indices])
    membership_guess = estimate_membership(
      gaia_data['pmra'][membership_pool],
      gaia_data['pmdec'][membership_pool],
      gaia_data['parallax'][membership_pool],
      gaia_data['r_med_geo'][membership_pool],
      gaia_data['r_lo_geo'][membership_pool],
      gaia_data['r_hi_geo'][membership_pool],
    )
    self.set_operation_progress(HRDiagram.PROGRESS_STEPS['GAIA_MATCH_DONE'])

    def gaia_value(column, gaia_index):
      # non-finite (masked) Gaia values become None to stay JSON-safe
      value = float(gaia_data[column][gaia_index])
      return round(value, 4) if np.isfinite(value) else None

    def gaia_astrometry(gaia_index):
      # proper motion, parallax, Bailer-Jones geometric distance, and the gaia_match flag for a
      # star (r_med_geo estimate + r_lo_geo/r_hi_geo 16th/84th percentile bounds.
      # The astrometry fields are None when the star has no match.
      matched = gaia_index is not None
      def value(column):
        return gaia_value(column, gaia_index) if matched else None
      return {
        'pmra': value('pmra'),
        'pmra_err': value('pmra_error'),
        'pmdec': value('pmdec'),
        'pmdec_err': value('pmdec_error'),
        'parallax': value('parallax'),
        'parallax_err': value('parallax_error'),
        'distance': value('r_med_geo'),
        'distance_lo': value('r_lo_geo'),
        'distance_hi': value('r_hi_geo'),
        'gaia_match': matched,
      }

    cmd_points = []
    # Iterate over points found in our input images and fill in gaia extra details
    for i in range(len(ra)):
      point = {
        'ra': round(float(ra[i]), 6),
        'dec': round(float(dec[i]), 6),
        'color': round(float(blue_mag[i] - red_mag[i]), 4),
        'color_err': round(float(np.sqrt(blue_magerr[i] ** 2 + red_magerr[i] ** 2)), 4),
        'mag': round(float(red_mag[i]), 4),
        'magerr': round(float(red_magerr[i]), 4),
      }
      point.update(gaia_astrometry(gaia_for_star.get(i)))
      point['gaia_only'] = False
      cmd_points.append(point)

    n_image_stars = len(cmd_points)
    # Gaia-only stars: no image detection, but their astrometry feeds the membership plots
    # and Gaia's synthetic photometry (where the source has it) also places them on the CMD
    blue_synth = HRDiagram.GAIA_SYNTH_MAG[blue_filter]
    red_synth = HRDiagram.GAIA_SYNTH_MAG[red_filter]
    for gaia_index in gaia_only_indices:
      # synthetic blue/red mags only feed color and the y-axis mag; they are not emitted per-star
      blue_synth_mag = gaia_value(f'{blue_synth}_mag', gaia_index)
      red_synth_mag = gaia_value(f'{red_synth}_mag', gaia_index)
      blue_err = gaia_value(f'{blue_synth}_mag_error', gaia_index)
      red_err = gaia_value(f'{red_synth}_mag_error', gaia_index)
      point = {
        'ra': round(float(gaia_data['ra'][gaia_index]), 6),
        'dec': round(float(gaia_data['dec'][gaia_index]), 6),
        'g_mag': gaia_value('phot_g_mean_mag', gaia_index),
        'mag': red_synth_mag,
        'magerr': red_err,
        'gaia_only': True,
      }
      point.update(gaia_astrometry(gaia_index))
      if blue_synth_mag is not None and red_synth_mag is not None:
        point['color'] = round(blue_synth_mag - red_synth_mag, 4)
        point['color_err'] = round(float(np.hypot(blue_err, red_err)), 4) \
          if blue_err is not None and red_err is not None else None
      else:
        point['color'] = None
        point['color_err'] = None
      cmd_points.append(point)

    output = {
      'output_data': [
        {
          'cluster': {'name': cluster_name, 'ra': cluster_ra, 'dec': cluster_dec, 'radius_arcmin': search_radius_arcmin},
          'blue_filter': blue_filter,
          'red_filter': red_filter,
          'mag_band': red_filter,
          'cmd': cmd_points,
          'membership_guess': membership_guess,
          'n_gaia_matched': len(star_indices),
          'n_gaia_only': len(gaia_only_indices),
          'n_stars': n_image_stars,
          'n_stars_matched': n_stars_matched,
        }
      ]
    }
    log.info(f'HR Diagram output {n_image_stars} image stars from {len(blue_indices)} band matches, '
             f'{len(star_indices)} Gaia matched, plus {len(gaia_only_indices)} Gaia-only stars')

    self.set_output(output, is_raw=True)
    self.set_operation_progress(HRDiagram.PROGRESS_STEPS['OUTPUT_DONE'])
    self.set_status('COMPLETED')
