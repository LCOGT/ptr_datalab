from datalab.datalab_session.utils.file_utils import *
from datalab.datalab_session.utils.s3_utils import *
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag, flux_to_mag_array, flux_to_mag_scalar
from datalab.datalab_session.utils.geometry import angular_distance_arcsec, distance_pixels
from datalab.datalab_session.utils.centroiding import BackgroundModel, _fit_plane
from datalab.datalab_session.utils.photometry import fractional_pixel_overlap, measure_aperture
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
