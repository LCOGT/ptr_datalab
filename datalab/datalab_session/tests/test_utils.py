import numpy as np
import pytest
from datalab.datalab_session.util import scale_points, stack_arrays

def test_points_scale_points():
    small_img_width = 200
    small_img_height = 200
    img_array = np.zeros((400, 400))
    points = [(10, 20), (30, 40), (50, 60), (0, 0)]
    expected_scaled_points = [(20, 40), (60, 80), (100, 120), (0, 0)]

    scaled_points = scale_points(small_img_width, small_img_height, img_array, points)

    assert len(scaled_points) == len(points)

    for i in range(len(scaled_points)):
      assert all([a == b for a, b in zip(scaled_points[i], expected_scaled_points[i])])

def test_scale_points_aspect_check_fails():
    small_img_width = 200
    small_img_height = 200
    img_array = np.zeros((400, 300))
    points = [(10, 20), (30, 40), (50, 60), (0, 0)]

    with pytest.raises(ValueError):
        scale_points(small_img_width, small_img_height, img_array, points)

def test_stack_arrays_equal_shape():
  array_list = [np.zeros((10, 10)), np.ones((10, 10)), np.full((10, 10), 2)]
  expected_shape = (10, 10, 3)
  stacked = stack_arrays(array_list)
  assert stacked.shape == expected_shape

def test_stack_arrays_cropped_data():
  array_list = [np.zeros((10, 10)), np.ones((15, 15)), np.full((12, 12), 2)]
  expected_shape = (10, 10, 3)
  stacked = stack_arrays(array_list)
  assert stacked.shape == expected_shape

def test_stack_arrays_empty_list():
  array_list = []
  with pytest.raises(ValueError):
    stack_arrays(array_list)

def test_stack_arrays_single_array():
  array_list = [np.zeros((10, 10))]
  expected_shape = (10, 10, 1)
  stacked = stack_arrays(array_list)
  assert stacked.shape == expected_shape
