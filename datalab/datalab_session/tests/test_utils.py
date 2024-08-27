import pathlib as pl
from hashlib import md5

from django.test import TestCase

from datalab.datalab_session.file_utils import *
from datalab.datalab_session.s3_utils import *

# extending the TestCase class to include a custom assertions for file operations
class FileExtendedTestCase(TestCase):
    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))
    
    def assertFilesEqual(self, image_1: str, image_2: str):
        with open(image_1, 'rb') as file_1, open(image_2, 'rb') as file_2:
            self.assertEqual(md5(file_1.read()).hexdigest(), md5(file_2.read()).hexdigest())

class FileUtilsTestClass(FileExtendedTestCase):

  test_fits = 'datalab/datalab_session/tests/test_files/file_utils/fits_1.fits.fz'
  test_tif = 'datalab/datalab_session/tests/test_files/file_utils/tif_1.tif'
  test_small_jpg = 'datalab/datalab_session/tests/test_files/file_utils/jpg_small_1.jpg'
  test_large_jpg = 'datalab/datalab_session/tests/test_files/file_utils/jpg_large_1.jpg'

  def test_get_fits_dimensions(self):
    fits_path = self.test_fits
    self.assertEqual(get_fits_dimensions(fits_path), (100, 100))

  def test_create_fits(self):
    test_2d_ndarray = np.zeros((10, 10))
    path = create_fits('create_fits_test', test_2d_ndarray)

    # test the file was written out to a path
    self.assertIsInstance(path, str)
    self.assertIsFile(path)
    
    # test the file has the right data
    hdu = fits.open(path)
    self.assertEqual(hdu[0].header['KEY'], 'create_fits_test')
    self.assertEqual(hdu[1].data.tolist(), test_2d_ndarray.tolist())
  
  def test_create_tif(self):
    fits_path = self.test_fits
    tif_path = create_tif('create_tif_test', fits_path)

    # test the file was written out to a path
    self.assertIsInstance(tif_path, str)
    self.assertIsFile(tif_path)

    self.assertFilesEqual(tif_path, self.test_tif)
  
  def test_create_jpgs(self):
    fits_path = self.test_fits
    jpg_paths = create_jpgs('create_jpgs_test', fits_path)

    # test the files were written out to a path
    self.assertEqual(len(jpg_paths), 2)
    self.assertIsFile(jpg_paths[0])
    self.assertIsFile(jpg_paths[1])
    self.assertFilesEqual(jpg_paths[0], self.test_large_jpg)
    self.assertFilesEqual(jpg_paths[1], self.test_small_jpg)
  
  def test_stack_arrays(self):
    test_array_1 = np.zeros((10, 20))
    test_array_2 = np.ones((20, 10))

    stacked_array = stack_arrays([test_array_1, test_array_2])
    self.assertIsInstance(stacked_array, np.ndarray)
    self.assertEqual(stacked_array.shape, (10, 10, 2))
    self.assertEqual(stacked_array[:, :, 0].tolist(), np.zeros((10, 10)).tolist())
    self.assertEqual(stacked_array[:, :, 1].tolist(), np.ones((10, 10)).tolist())

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
