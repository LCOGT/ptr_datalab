from unittest import mock
import json

from django.test import TestCase
from numpy.testing import assert_almost_equal

from datalab.datalab_session.analysis.line_profile import line_profile
from datalab.datalab_session.analysis.source_catalog import source_catalog

class TestAnalysis(TestCase):
    analysis_test_path = 'datalab/datalab_session/tests/test_files/analysis/'
    analysis_fits_1_path = f'datalab/datalab_session/tests/test_files/fits_1.fits.fz'

    def setUp(self):
        with open(f'{self.analysis_test_path}test_line_profile.json') as f:
            self.test_line_profile_data = json.load(f)['test_line_profile']
        
        with open(f'{self.analysis_test_path}test_source_catalog.json') as f:
            self.test_source_catalog_data = json.load(f)['test_source_catalog']
    
    @mock.patch('datalab.datalab_session.analysis.line_profile.FileCache')
    def test_line_profile(self, mock_file_cache):
        mock_instance = mock_file_cache.return_value
        mock_instance.get_fits.return_value = self.analysis_fits_1_path

        output = line_profile({
            'basename': 'fits_1',
            'height': 100,
            'width': 100,
            'x1': 25,
            'y1': 25,
            'x2': 75,
            'y2': 75,
            'source': 'archive'
        }, None)

        assert_almost_equal(output.get('line_profile').tolist(), self.test_line_profile_data, decimal=3)

    @mock.patch('datalab.datalab_session.analysis.source_catalog.FileCache')
    def test_source_catalog(self, mock_file_cache):
        mock_instance = mock_file_cache.return_value

        mock_instance.get_fits.return_value = self.analysis_fits_1_path
        
        output = source_catalog({
            'basename': 'fits_1',
            'height': 100,
            'width': 100,
            'source': 'archive'
            }, None)

        self.assertEqual(output, self.test_source_catalog_data)
