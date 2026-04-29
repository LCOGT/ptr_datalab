from unittest import mock
import json
from types import SimpleNamespace

from astropy.io import fits
from django.test import TestCase
import numpy as np
from numpy.testing import assert_almost_equal

from datalab.datalab_session.analysis import centroiding, line_profile, source_catalog
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag

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

        output = line_profile.line_profile({
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
        
        output = source_catalog.source_catalog({
            'basename': 'fits_1',
            'height': 100,
            'width': 100,
            'source': 'archive'
            }, None)

        self.assertEqual(len(output), len(self.test_source_catalog_data))

        for result, expected in zip(output, self.test_source_catalog_data):
            self.assertEqual(result['x_win'], expected['x_win'])
            self.assertEqual(result['y_win'], expected['y_win'])
            self.assertEqual(result['x'], expected['x'])
            self.assertEqual(result['y'], expected['y'])
            self.assertEqual(result['flux'], expected['flux'])
            self.assertEqual(result['ra'], expected['ra'])
            self.assertEqual(result['dec'], expected['dec'])
            self.assertIn('mag', result)
            self.assertIn('magerr', result)

        with fits.open(self.analysis_fits_1_path) as hdul:
            fluxerr = hdul['CAT'].data['fluxerr'][0]

        expected_mag, expected_magerr = flux_to_mag(output[0]['flux'], fluxerr)
        self.assertAlmostEqual(output[0]['mag'], expected_mag)
        self.assertAlmostEqual(output[0]['magerr'], expected_magerr)

    def test_centroid_finds_pixels_center(self):
        image = np.zeros((21, 21), dtype=float)
        image[10, 10] = 100.0

        result = centroiding.centroid(
            image,
            x_click=10.0,
            y_click=10.0,
            radius=3.0,
            r_back1=4.0,
            r_back2=5.0,
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x, 10.5, places=9)
        self.assertAlmostEqual(result.y, 10.5, places=9)
        self.assertEqual(result.background, 0.0)
        self.assertEqual(result.peak, 100.0)
        self.assertEqual(result.message, 'Centroid calculation completed.')

    @mock.patch('datalab.datalab_session.analysis.centroiding.get_hdu')
    @mock.patch('datalab.datalab_session.analysis.centroiding.FileCache')
    def test_centroiding_scales_display_coordinates(self, mock_file_cache, mock_get_hdu):
        mock_instance = mock_file_cache.return_value
        mock_instance.get_fits.return_value = self.analysis_fits_1_path

        fits_image = np.zeros((80, 120), dtype=float)
        fits_image[48, 36] = 1200.0
        mock_get_hdu.return_value = SimpleNamespace(data=fits_image)
        input_data = {
            'basename': 'fits_1',
            'height': 160,
            'width': 240,
            'x': 72.0,
            'y': 96.0,
            'radius': 3.0,
            'r_back1': 4.0,
            'r_back2': 5.0,
            'source': 'archive',
        }

        output = centroiding.centroiding(input_data, None)

        self.assertTrue(output['success'])
        self.assertAlmostEqual(output['x'], 73.0, places=9)
        self.assertAlmostEqual(output['y'], 97.0, places=9)
        self.assertEqual(output['background'], 0.0)
        self.assertEqual(output['peak'], 1200.0)
        self.assertEqual(output['message'], 'Centroid calculation completed.')
        self.assertIsNone(output['ra'])
        self.assertIsNone(output['dec'])

    @mock.patch('datalab.datalab_session.analysis.centroiding.get_hdu')
    @mock.patch('datalab.datalab_session.analysis.centroiding.FileCache')
    def test_centroiding_returns_ra_dec(self, mock_file_cache, mock_get_hdu):
        mock_instance = mock_file_cache.return_value
        mock_instance.get_fits.return_value = self.analysis_fits_1_path

        fits_image = np.zeros((80, 120), dtype=float)
        fits_image[48, 36] = 1200.0
        header = fits.Header()
        header['CTYPE1'] = 'RA---TAN'
        header['CTYPE2'] = 'DEC--TAN'
        header['CRVAL1'] = 150.0
        header['CRVAL2'] = 2.0
        header['CRPIX1'] = 1.0
        header['CRPIX2'] = 1.0
        header['CD1_1'] = 0.01
        header['CD1_2'] = 0.0
        header['CD2_1'] = 0.0
        header['CD2_2'] = 0.01
        mock_get_hdu.return_value = SimpleNamespace(data=fits_image, header=header)
        input_data = {
            'basename': 'fits_1',
            'height': 160,
            'width': 240,
            'x': 72.0,
            'y': 96.0,
            'radius': 3.0,
            'r_back1': 4.0,
            'r_back2': 5.0,
            'source': 'archive',
        }

        output = centroiding.centroiding(input_data, None)

        self.assertTrue(output['success'])
        self.assertAlmostEqual(output['x'], 73.0, places=9)
        self.assertAlmostEqual(output['y'], 97.0, places=9)
        self.assertEqual(output['message'], 'Centroid calculation completed.')
        self.assertIsNotNone(output['ra'])
        self.assertIsNotNone(output['dec'])

    def test_centroid_returns_message_when_no_valid_pixels_in_box(self):
        image = np.full((21, 21), np.nan, dtype=float)

        result = centroiding.centroid(
            image,
            x_click=10.0,
            y_click=10.0,
            radius=3.0,
            r_back1=4.0,
            r_back2=5.0,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.message, 'No valid pixels in centroid box.')

    def test_centroid_returns_message_when_zero_weight(self):
        image = np.zeros((21, 21), dtype=float)

        result = centroiding.centroid(
            image,
            x_click=10.0,
            y_click=10.0,
            radius=3.0,
            r_back1=4.0,
            r_back2=5.0,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.message, 'Centroid calculation has zero weight in both dimensions.')
