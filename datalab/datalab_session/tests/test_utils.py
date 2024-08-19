import pathlib as pl

from django.test import TestCase

from datalab.datalab_session.util import *
from django.conf import settings

# extending the TestCase class to include a custom assertions
class TestCaseCustom(TestCase):
    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

class S3UtilTestClass(TestCase):

  def test_bucket(self):
    # Test that the bucket is set to the correct value
    self.assertEqual(settings.DATALAB_OPERATION_BUCKET, 'datalab-operation-output-lco-global')

  def test_key_exists(self):
    self.assertFalse(key_exists('nonexistent_test_key'))

class FitsUtilTestClass(TestCaseCustom):

  def test_get_fits_dimensions(self):
    fits_path = 'datalab/datalab_session/tests/test_files/test.fits'
    self.assertEqual(get_fits_dimensions(fits_path), (2400, 2400))

  def test_create_fits(self):
    test_2d_ndarray = np.zeros((10, 10))
    path = create_fits('create_fits_test', test_2d_ndarray)
    self.assertIsInstance(path, str)
    self.assertIsFile(path)
  
  def test_stack_arrays(self):
    test_array_1 = np.zeros((10, 20))
    test_array_2 = np.ones((20, 10))

    stacked_array = stack_arrays([test_array_1, test_array_2])
    self.assertIsInstance(stacked_array, np.ndarray)
    self.assertEqual(stacked_array.shape, (10, 10, 2))

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
