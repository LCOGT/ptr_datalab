from datalab.datalab_session.data_operations.median import Median

class TestMedianOperation:
  def test_generate_cache_key(self):
    input_data = {'key1': 'value1', 'key2': 'value2'}
    data_operation = Median(input_data)
    cache_key = data_operation.generate_cache_key()
    assert isinstance(cache_key, str)
    assert len(cache_key) > 0

  def test_normalize_input_data_with_none_input(self):
    input_data = None
    data_operation = Median(input_data)
    normalized_data = data_operation._normalize_input_data(input_data)
    assert normalized_data == {}

  def test_normalize_input_data_with_empty_input(self):
    input_data = {}
    data_operation = Median(input_data)
    normalized_data = data_operation._normalize_input_data(input_data)
    assert normalized_data == {}

  def test_normalize_input_data_with_file_inputs(self):
    input_data = {
      'file1': [{'basename': 'file1.txt'}, {'basename': 'file2.txt'}],
      'file2': [{'basename': 'file3.txt'}, {'basename': 'file4.txt'}]
    }
    data_operation = Median(input_data)
    normalized_data = data_operation._normalize_input_data(input_data)
    expected_data = {
      'file1': [{'basename': 'file1.txt'}, {'basename': 'file2.txt'}],
      'file2': [{'basename': 'file3.txt'}, {'basename': 'file4.txt'}]
    }
    assert normalized_data == expected_data

  def test_normalize_input_data_with_sorted_file_inputs(self):
    input_data = {
      'file1': [{'basename': 'file4.txt'}],
      'file2': [{'basename': 'file1.txt'}]
    }
    data_operation = Median(input_data)
    normalized_data = data_operation._normalize_input_data(input_data)
    expected_data = {
      'file2': [{'basename': 'file1.txt'}],
      'file1': [{'basename': 'file4.txt'}]
    }
    assert normalized_data == expected_data

  def test_name_is_median(self):
    assert Median.name() == 'Median'
