from unittest import mock

from astropy.table import Table, MaskedColumn

from datalab.datalab_session.utils.file_utils import *
from datalab.datalab_session.utils.s3_utils import *
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag, flux_to_mag_array, flux_to_mag_scalar
from datalab.datalab_session.utils.geometry import angular_distance_arcsec, distance_pixels
from datalab.datalab_session.utils.centroiding import BackgroundModel, _fit_plane
from datalab.datalab_session.utils.photometry import fractional_pixel_overlap, measure_aperture
from datalab.datalab_session.utils.catalog_utils import (cross_match_one_to_one, cone_filter, find_nearest_source,
                                                         mean_observation_epoch, propagate_positions)
from datalab.datalab_session.utils.gaia import (GAIA_EPOCH, GAIA_PARALLAX_ZERO_POINT_MAS, estimate_membership,
                                                gaia_cone_search)
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.tests.test_files.file_extended_test_case import FileExtendedTestCase

class FileUtilsTestClass(FileExtendedTestCase):

  util_test_path = 'datalab/datalab_session/tests/test_files/file_utils/'

  test_fits_path = f'datalab/datalab_session/tests/test_files/fits_1.fits.fz'
  test_tif_path = f'{util_test_path}tif_1.tif'
  test_small_jpg_path = f'{util_test_path}jpg_small_1.jpg'
  test_large_jpg_path = f'{util_test_path}jpg_large_1.jpg'

  def test_get_fits_dimensions(self):
    fits_path = self.test_fits_path
    self.assertEqual(get_fits_dimensions(fits_path), (100, 100))

  def test_create_fits(self):
    test_2d_ndarray = np.zeros((10, 10))
    with create_fits('create_fits_test', test_2d_ndarray) as path:
      # test the file was written out to a path
      self.assertIsInstance(path, str)
      self.assertIsFile(path)
      
      # test the file has the right data
      hdu = fits.open(path)
      self.assertEqual(hdu[0].header['KEY'], 'create_fits_test')
      self.assertEqual(hdu[1].data.tolist(), test_2d_ndarray.tolist())

  def test_create_tif(self):
    fits_path = self.test_fits_path
    with temp_file_manager('test.tif') as tif_path:
      create_tif(fits_path, tif_path)
      self.assertIsInstance(tif_path, str)
      self.assertIsFile(tif_path)
      self.assertFilesEqual(tif_path, self.test_tif_path)

  def test_create_jpgs(self):
    fits_path = self.test_fits_path
    with temp_file_manager('large.jpg', 'small.jpg') as (large_jpg, small_jpg):
      create_jpgs(fits_path, large_jpg, small_jpg)
      self.assertIsFile(large_jpg)
      self.assertIsFile(small_jpg)
      self.assertFilesEqual(large_jpg, self.test_large_jpg_path)
      self.assertFilesEqual(small_jpg, self.test_small_jpg_path)
  
  def test_stack_arrays(self):
    test_array_1 = np.zeros((10, 20))
    test_array_2 = np.ones((20, 10))

    cropped_array, size = crop_arrays([test_array_1, test_array_2])
    self.assertEqual(len(cropped_array), 2)
    self.assertEqual(size, (10, 10))
    self.assertEqual(cropped_array[0].tolist(), np.zeros((10, 10)).tolist())
    self.assertEqual(cropped_array[1].tolist(), np.ones((10, 10)).tolist())

  def test_scale_points(self):
    x_points = [1, 2, 3]
    y_points = [4, 5, 6]

    # no flip
    scaled_points = scale_points(10, 10, 20, 20, x_points, y_points)
    self.assertEqual(len(scaled_points), 2)
    self.assertEqual(len(scaled_points[0]), 3)
    self.assertEqual(len(scaled_points[1]), 3)
    self.assertEqual(scaled_points[0].tolist(), [2, 4, 6])
    self.assertEqual(scaled_points[1].tolist(), [8, 10, 12])

    # flip x
    scaled_points = scale_points(10, 10, 20, 20, x_points, y_points, flip_x=True)
    self.assertEqual(scaled_points[0].tolist(), [18, 16, 14])

    # flip y
    scaled_points = scale_points(10, 10, 20, 20, x_points, y_points, flip_y=True)
    self.assertEqual(scaled_points[1].tolist(), [12, 10, 8])

  def test_flux_to_mag(self):
    mag, magerr = flux_to_mag(100.0, 5.0)

    self.assertAlmostEqual(mag, -5.0)
    self.assertAlmostEqual(magerr, 0.05428681023790647)

  def test_flux_to_mag_scalar(self):
    mag, magerr = flux_to_mag_scalar(100.0, 5.0)

    self.assertAlmostEqual(mag, -5.0)
    self.assertAlmostEqual(magerr, 0.05428681023790647)

  def test_flux_to_mag_rejects_non_positive_flux(self):
    self.assertEqual(flux_to_mag(0.0, 5.0), (None, None))
    self.assertEqual(flux_to_mag(-1.0, 5.0), (None, None))

  def test_flux_to_mag_supports_arrays(self):
    mag, magerr = flux_to_mag(np.array([100.0, 0.0]), np.array([5.0, 1.0]))

    self.assertAlmostEqual(mag[0], -5.0)
    self.assertAlmostEqual(magerr[0], 0.05428681023790647)
    self.assertTrue(np.isnan(mag[1]))
    self.assertTrue(np.isnan(magerr[1]))

  def test_flux_to_mag_array(self):
    mag, magerr = flux_to_mag_array(np.array([100.0, 0.0]), np.array([5.0, 1.0]))

    self.assertAlmostEqual(mag[0], -5.0)
    self.assertAlmostEqual(magerr[0], 0.05428681023790647)
    self.assertTrue(np.isnan(mag[1]))
    self.assertTrue(np.isnan(magerr[1]))

  def test_geometry_distance_helpers(self):
    self.assertEqual(distance_pixels(0.0, 0.0, 3.0, 4.0), 5.0)
    self.assertAlmostEqual(angular_distance_arcsec(10.0, 20.0, 10.0, 20.001), 3.6, places=3)

  def test_fractional_pixel_overlap(self):
    self.assertEqual(fractional_pixel_overlap(5, 5, 5.5, 5.5, 1.0), 1.0)
    self.assertEqual(fractional_pixel_overlap(8, 8, 5.5, 5.5, 1.0), 0.0)

  def test_measure_aperture(self):
    image = np.full((21, 21), 10.0, dtype=float)
    image[10, 10] = 110.0

    result = measure_aperture(
      image=image,
      x_center=10.5,
      y_center=10.5,
      aperture_radius_px=2.0,
      background_model=BackgroundModel(
        mean=10.0,
        effective_pixels=64.0,
      ),
      gain=1.0,
      read_noise=0.0,
      dark=0.0,
    )

    self.assertGreater(result["net_source_counts"], 0.0)
    self.assertEqual(result["mean_background_per_pixel"], 10.0)
    self.assertEqual(result["peak_pixel_value"], 110.0)
    self.assertEqual(result["effective_background_pixels"], 64.0)

  def test_fit_plane_accepts_point_tuples(self):
    points = [
      (0.0, 0.0, 4.0),
      (1.0, 0.0, 6.0),
      (0.0, 1.0, 1.0),
      (2.0, 1.0, 5.0),
      (-1.0, 2.0, -4.0),
    ]

    plane = _fit_plane(points)

    self.assertIsNotNone(plane)
    assert plane is not None
    self.assertAlmostEqual(plane.c0, 4.0)
    self.assertAlmostEqual(plane.c1, 2.0)
    self.assertAlmostEqual(plane.c2, -3.0)

  def test_fit_plane_rejects_degenerate_points(self):
    points = [
      (0.0, 0.0, 4.0),
      (1.0, 1.0, 5.0),
      (2.0, 2.0, 6.0),
      (3.0, 3.0, 7.0),
    ]

    self.assertIsNone(_fit_plane(points))

class CatalogUtilsTestClass(FileExtendedTestCase):

  epoch_test_path = 'datalab/datalab_session/tests/test_files/temp_epoch_{}.fits'

  def tearDown(self):
    self.clean_test_dir()
    return super().tearDown()

  @staticmethod
  def catalog(dec_offsets_arcsec, ra=150.0, dec=30.0):
    """ Builds a minimal catalog dict of points offset north of (ra, dec) by arcseconds """
    return {
      'ra': np.full(len(dec_offsets_arcsec), ra),
      'dec': dec + np.array(dec_offsets_arcsec) / 3600.0,
    }

  def write_dated_fits(self, path, date_obs):
    sci_hdu = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32), header=fits.Header({'DATE-OBS': date_obs}),
                            name='SCI')
    fits.HDUList([fits.PrimaryHDU(), sci_hdu]).writeto(path, overwrite=True)
    return path

  def test_mean_observation_epoch(self):
    # midway between the two exposures, half a year into 2026
    paths = [self.write_dated_fits(self.epoch_test_path.format(index), date_obs)
             for index, date_obs in enumerate(['2026-01-01T00:00:00.000', '2027-01-01T00:00:00.000'])]

    self.assertAlmostEqual(mean_observation_epoch(paths), 2026.5, places=3)
    self.assertAlmostEqual(mean_observation_epoch(paths[:1]), 2026.0, places=3)

  @staticmethod
  def catalog_columns(*columns):
    """ propagate_positions takes the float arrays a catalog is read into, as gaia.py returns """
    return [np.array(column, dtype=float) for column in columns]

  def test_propagate_positions_moves_stars_along_proper_motion(self):
    # 3600 mas/yr for 10 years is exactly 36 arcsec of great-circle motion in each axis
    columns = self.catalog_columns([150.0], [30.0], [3600.0], [3600.0])

    ra, dec = propagate_positions(*columns, 2016.0, 2026.0)

    self.assertAlmostEqual(dec[0], 30.0 + 36.0 / 3600.0, places=9)
    # pmra is pmra*, so the same angle covers more degrees of RA away from the equator
    self.assertAlmostEqual(ra[0], 150.0 + (36.0 / 3600.0) / np.cos(np.radians(30.0)), places=9)

  def test_propagate_positions_keeps_stars_without_proper_motion(self):
    columns = self.catalog_columns([150.0, 151.0], [30.0, 31.0], [np.nan, 100.0], [np.nan, np.nan])

    ra, dec = propagate_positions(*columns, 2016.0, 2026.0)

    self.assertAlmostEqual(ra[0], 150.0)
    self.assertAlmostEqual(dec[0], 30.0)
    # a partial solution still moves in the axis it has
    self.assertGreater(ra[1], 151.0)
    self.assertAlmostEqual(dec[1], 31.0)

  def test_cross_match_matches_within_radius(self):
    catalog_a = self.catalog([0.0, 10.0])
    catalog_b = self.catalog([0.5, 10.0])

    a_indices, b_indices = cross_match_one_to_one(catalog_a, catalog_b, 2.0)

    self.assertEqual(a_indices.tolist(), [0, 1])
    self.assertEqual(b_indices.tolist(), [0, 1])

  def test_cross_match_applies_separation_cut(self):
    catalog_a = self.catalog([0.0])
    catalog_b = self.catalog([3.0])  # nearest neighbor is 3 arcsec away

    a_indices, b_indices = cross_match_one_to_one(catalog_a, catalog_b, 2.0)

    self.assertEqual(len(a_indices), 0)
    self.assertEqual(len(b_indices), 0)

  def test_cross_match_dedups_to_closest_pair(self):
    # two a sources share the same nearest b source, only the closer pair survives
    catalog_a = self.catalog([0.0, 1.0])
    catalog_b = self.catalog([0.0])

    a_indices, b_indices = cross_match_one_to_one(catalog_a, catalog_b, 2.0)

    self.assertEqual(a_indices.tolist(), [0])
    self.assertEqual(b_indices.tolist(), [0])

  def test_cross_match_empty_catalogs(self):
    empty = {'ra': np.array([]), 'dec': np.array([])}
    catalog = self.catalog([0.0])

    for catalog_a, catalog_b in [(empty, catalog), (catalog, empty), (empty, empty)]:
      a_indices, b_indices = cross_match_one_to_one(catalog_a, catalog_b, 2.0)
      self.assertEqual(len(a_indices), 0)
      self.assertEqual(len(b_indices), 0)

  def test_cone_filter(self):
    catalog = self.catalog([30.0, 120.0])  # 0.5 and 2 arcmin north of center

    in_cone = cone_filter(catalog['ra'], catalog['dec'], 150.0, 30.0, 1.0)

    self.assertEqual(in_cone.tolist(), [True, False])

  def test_find_nearest_source_returns_closest_not_first(self):
    # a neighbor is listed first, the real target second: catalogs are ordered by detection,
    # not by distance from the target
    catalog = self.catalog([3.0, 0.2])

    nearest = find_nearest_source(catalog['ra'], catalog['dec'], 150.0, 30.0, 4.0)

    self.assertEqual(nearest, 1)

  def test_find_nearest_source_applies_radius_cut(self):
    catalog = self.catalog([5.0])

    self.assertIsNone(find_nearest_source(catalog['ra'], catalog['dec'], 150.0, 30.0, 4.0))

  def test_find_nearest_source_handles_ra_wraparound(self):
    # 3.55 arcsec apart across the 0/360 boundary, which a coordinate difference reads as 359.999
    nearest = find_nearest_source(np.array([0.0005]), np.array([10.0]), 359.9995, 10.0, 4.0)

    self.assertEqual(nearest, 0)

  def test_find_nearest_source_tolerance_holds_at_high_declination(self):
    # 3 arcsec away on sky, but 6 arcsec of RA coordinate at this declination
    dec = 60.0
    ra_offset = (3.0 / 3600.0) / np.cos(np.radians(dec))

    nearest = find_nearest_source(np.array([150.0 + ra_offset]), np.array([dec]), 150.0, dec, 4.0)

    self.assertEqual(nearest, 0)

  def test_find_nearest_source_empty_catalog(self):
    self.assertIsNone(find_nearest_source(np.array([]), np.array([]), 150.0, 30.0, 4.0))


class GaiaUtilsTestClass(FileExtendedTestCase):

  @staticmethod
  def gaia_table():
    """ A fake Gaia archive result: the second source is missing its astrometric solution,
    its Bailer-Jones distance, and (like any source without published BP/RP spectra) its
    synthetic photometry """
    columns = {
      'ra': [150.0, 150.001],
      'dec': [30.0, 30.001],
      'pmra': MaskedColumn([5.0, 0.0], mask=[False, True]),
      'pmra_error': MaskedColumn([0.05, 0.0], mask=[False, True]),
      'pmdec': MaskedColumn([-3.0, 0.0], mask=[False, True]),
      'pmdec_error': MaskedColumn([0.06, 0.0], mask=[False, True]),
      'parallax': MaskedColumn([1.25, 0.0], mask=[False, True]),
      'parallax_error': MaskedColumn([0.04, 0.0], mask=[False, True]),
      'phot_g_mean_mag': [14.2, 17.9],
      'r_med_geo': MaskedColumn([800.0, 0.0], mask=[False, True]),
      'r_lo_geo': MaskedColumn([760.0, 0.0], mask=[False, True]),
      'r_hi_geo': MaskedColumn([845.0, 0.0], mask=[False, True]),
    }
    for band in ['g_sdss', 'r_sdss', 'i_sdss', 'z_sdss']:
      columns[f'{band}_mag'] = MaskedColumn([15.0, 0.0], mask=[False, True])
      columns[f'{band}_flux'] = MaskedColumn([1000.0, 0.0], mask=[False, True])
      columns[f'{band}_flux_error'] = MaskedColumn([10.0, 0.0], mask=[False, True])
    return Table(columns)

  @mock.patch('astroquery.gaia.Gaia')
  def test_gaia_cone_search_parses_and_caches(self, mock_gaia):
    mock_gaia.launch_job_async.return_value.get_results.return_value = self.gaia_table()

    # unique field so this test never hits another test's cache entry
    gaia_data = gaia_cone_search(210.1234, -5.5678, 12.0)

    self.assertEqual(len(gaia_data['ra']), 2)
    self.assertAlmostEqual(gaia_data['pmra'][0], 5.0)
    self.assertAlmostEqual(gaia_data['parallax'][0], 1.25 + GAIA_PARALLAX_ZERO_POINT_MAS)
    # masked values come back as nan
    self.assertTrue(np.isnan(gaia_data['pmra'][1]))
    self.assertTrue(np.isnan(gaia_data['parallax'][1]))
    # Bailer-Jones geometric distance parses; the source without an entry comes back nan
    self.assertAlmostEqual(gaia_data['r_med_geo'][0], 800.0)
    self.assertAlmostEqual(gaia_data['r_lo_geo'][0], 760.0)
    self.assertAlmostEqual(gaia_data['r_hi_geo'][0], 845.0)
    self.assertTrue(np.isnan(gaia_data['r_med_geo'][1]))
    # synthetic photometry parses, with flux errors propagated onto the systematic floor
    self.assertAlmostEqual(gaia_data['g_sdss_mag'][0], 15.0)
    expected_error = np.sqrt((2.5 / np.log(10) * 10.0 / 1000.0) ** 2 + 0.03 ** 2)
    self.assertAlmostEqual(gaia_data['g_sdss_mag_error'][0], expected_error, places=5)
    self.assertTrue(np.isnan(gaia_data['g_sdss_mag'][1]))
    self.assertTrue(np.isnan(gaia_data['g_sdss_mag_error'][1]))

    # a second identical search is served from the cache without re-querying the archive
    gaia_cone_search(210.1234, -5.5678, 12.0)
    mock_gaia.launch_job_async.assert_called_once()

  @mock.patch('astroquery.gaia.Gaia')
  def test_gaia_cone_search_failure_raises(self, mock_gaia):
    mock_gaia.launch_job_async.side_effect = ConnectionError('archive down')

    with self.assertRaisesRegex(ClientAlertException, 'Gaia'):
      gaia_cone_search(33.3333, 44.4444, 10.0)

  @staticmethod
  def cluster_in_field(rng, cluster_pmra, cluster_pmdec, n_cluster=40, n_field=160):
    """ A tight cluster clump in a broad field centered on zero proper motion, with the cluster
    nearby (7.4 mas parallax, ~135 pc) and the field scattered behind it """
    pmra = np.concatenate([rng.normal(cluster_pmra, 0.4, n_cluster), rng.normal(0.0, 4.0, n_field)])
    pmdec = np.concatenate([rng.normal(cluster_pmdec, 0.4, n_cluster), rng.normal(0.0, 4.0, n_field)])
    parallax = np.concatenate([rng.normal(7.4, 0.1, n_cluster), rng.uniform(0.1, 2.0, n_field)])
    distance = np.concatenate([rng.normal(135.0, 5.0, n_cluster), rng.uniform(500.0, 3000.0, n_field)])
    return pmra, pmdec, parallax, distance

  def test_estimate_membership_finds_clump_far_from_field_median(self):
    rng = np.random.default_rng(5)
    # a Pleiades-like cluster moving at 49 mas/yr, and only a fifth of the pool - a fixed window
    # around the field's median proper motion would bin it away entirely and lock onto the field
    pmra, pmdec, parallax, distance = self.cluster_in_field(rng, 20.0, -45.0)

    guess = estimate_membership(pmra, pmdec, parallax, distance, distance - 10.0, distance + 10.0)

    self.assertIsNotNone(guess)
    self.assertAlmostEqual(guess['pmra'], 20.0, delta=0.5)
    self.assertAlmostEqual(guess['pmdec'], -45.0, delta=0.5)
    # the windows follow the cluster, not the field it is embedded in
    self.assertLess(guess['parallax_min'], 7.4)
    self.assertGreater(guess['parallax_max'], 7.4)
    self.assertLess(guess['distance_min'], 135.0)
    self.assertGreater(guess['distance_max'], 135.0)

  def test_estimate_membership_tolerates_high_proper_motion_stray(self):
    rng = np.random.default_rng(5)
    # one foreground star crossing the sky at 8 arcsec/yr must not stretch the histogram over it
    pmra, pmdec, parallax, distance = self.cluster_in_field(rng, 20.0, -45.0)
    pmra, pmdec = np.append(pmra, 8000.0), np.append(pmdec, -3000.0)
    parallax, distance = np.append(parallax, 500.0), np.append(distance, 2.0)

    guess = estimate_membership(pmra, pmdec, parallax, distance, distance - 10.0, distance + 10.0)

    self.assertIsNotNone(guess)
    self.assertAlmostEqual(guess['pmra'], 20.0, delta=0.5)
    self.assertAlmostEqual(guess['pmdec'], -45.0, delta=0.5)

  def test_estimate_membership_finds_clump(self):
    rng = np.random.default_rng(42)
    # a 40 star cluster clump at (-5, 3) mas/yr among 20 spread out field stars
    pmra = np.concatenate([rng.normal(-5.0, 0.3, 40), rng.uniform(-20.0, 20.0, 20)])
    pmdec = np.concatenate([rng.normal(3.0, 0.3, 40), rng.uniform(-20.0, 20.0, 20)])
    parallax = np.concatenate([rng.normal(1.25, 0.05, 40), rng.uniform(0.1, 2.0, 20)])
    # cluster piled up near 800 pc, field stars scattered; +/-40 pc Bailer-Jones bounds per star
    distance = np.concatenate([rng.normal(800.0, 20.0, 40), rng.uniform(100.0, 3000.0, 20)])
    distance_lo = distance - 40.0
    distance_hi = distance + 40.0

    guess = estimate_membership(pmra, pmdec, parallax, distance, distance_lo, distance_hi)

    self.assertIsNotNone(guess)
    self.assertAlmostEqual(guess['pmra'], -5.0, delta=0.5)
    self.assertAlmostEqual(guess['pmdec'], 3.0, delta=0.5)
    self.assertGreaterEqual(guess['pm_radius'], 0.5)
    self.assertLessEqual(guess['pm_radius'], 5.0)
    self.assertLess(guess['parallax_min'], 1.25)
    self.assertGreater(guess['parallax_max'], 1.25)
    self.assertLess(guess['distance_min'], 800.0)
    self.assertGreater(guess['distance_max'], 800.0)
    self.assertGreaterEqual(guess['distance_min'], 0.0)

  def test_estimate_membership_without_member_distances(self):
    rng = np.random.default_rng(42)
    pmra = np.concatenate([rng.normal(-5.0, 0.3, 40), rng.uniform(-20.0, 20.0, 20)])
    pmdec = np.concatenate([rng.normal(3.0, 0.3, 40), rng.uniform(-20.0, 20.0, 20)])
    parallax = np.concatenate([rng.normal(1.25, 0.05, 40), rng.uniform(0.1, 2.0, 20)])
    # no source has a Bailer-Jones distance - the distance window is left unset
    distance = np.full(60, np.nan)

    guess = estimate_membership(pmra, pmdec, parallax, distance, distance, distance)

    self.assertIsNotNone(guess)
    self.assertIsNone(guess['distance_min'])
    self.assertIsNone(guess['distance_max'])

  def test_estimate_membership_too_few_stars(self):
    pmra = np.full(5, -5.0)
    pmdec = np.full(5, 3.0)
    parallax = np.full(5, 1.25)
    distance = np.full(5, 800.0)

    self.assertIsNone(estimate_membership(pmra, pmdec, parallax, distance, distance - 40.0, distance + 40.0))

  def test_estimate_membership_no_clump(self):
    rng = np.random.default_rng(7)
    # spread out field stars only, no concentration to call a cluster
    pmra = rng.uniform(-20.0, 20.0, 40)
    pmdec = rng.uniform(-20.0, 20.0, 40)
    parallax = rng.uniform(0.1, 2.0, 40)
    distance = rng.uniform(100.0, 3000.0, 40)

    self.assertIsNone(estimate_membership(pmra, pmdec, parallax, distance, distance - 40.0, distance + 40.0))

  def test_estimate_membership_tolerates_negative_parallax(self):
    rng = np.random.default_rng(11)
    pmra = rng.normal(0.0, 0.2, 15)
    pmdec = rng.normal(0.0, 0.2, 15)
    parallax = rng.normal(-0.05, 0.02, 15)  # a distant clump can measure slightly negative
    distance = rng.normal(15000.0, 500.0, 15)

    guess = estimate_membership(pmra, pmdec, parallax, distance, distance - 2000.0, distance + 2000.0)

    self.assertIsNotNone(guess)
    self.assertLess(guess['parallax_min'], 0.0)
