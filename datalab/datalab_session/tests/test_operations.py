from unittest import mock
from hashlib import md5
import os

from django.test import TestCase

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation

from datalab.datalab_session.data_operations.rgb_stack import RGB_Stack
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.data_operations.median import Median

wizard_description = {
            'name': 'SampleDataOperation',
            'description': 'Testing class for DataOperation',
            'category': 'test',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 999
                },
                'input_number': {
                    'name': 'Input Number',
                    'description': 'The input number to operate on',
                    'type': 'number',
                    'minimum': 1,
                    'maximum': 999
                },
                'input_string': {
                    'name': 'Input String',
                    'description': 'The input string to operate on',
                    'type': 'string',
                    'minimum': 1,
                    'maximum': 999
                }
            }
        }

class SampleDataOperation(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'SampleDataOperation'
    
    @staticmethod
    def description():
        return 'Testing class for DataOperation'
    
    @staticmethod
    def wizard_description():
        return wizard_description
    
    def operate(self):
        self.set_output({'output_files': []})

class TestDataOperation(TestCase):
    def setUp(self):
        input_data = {
            'input_files': [],
            'input_number': 1,
            'input_string': 'test'
        }
        self.data_operation = SampleDataOperation(input_data)

    def test_init(self):
        self.assertEqual(self.data_operation.input_data, {
            'input_files': [],
            'input_number': 1,
            'input_string': 'test'
        })
        self.assertIsInstance(self.data_operation.cache_key, str)
        self.assertGreater(len(self.data_operation.cache_key), 0)
    
    def test_normalize_input_data(self):
        # out of order files
        input_data = {
            'input_files': [
                {'basename': 'file2'},
                {'basename': 'file1'}
            ],
            'input_number': 1,
            'input_string': 'test'
        }
        normalized_input_data = self.data_operation._normalize_input_data(input_data)
        self.assertEqual(normalized_input_data, {
            'input_files': [
                {'basename': 'file1'},
                {'basename': 'file2'}
            ],
            'input_number': 1,
            'input_string': 'test'
        })

        # in order files
        input_data = {
            'input_files': [
                {'basename': 'file1'},
                {'basename': 'file2'}
            ],
            'input_number': 1,
            'input_string': 'test'
        }
        normalized_input_data = self.data_operation._normalize_input_data(input_data)
        self.assertEqual(normalized_input_data, {
            'input_files': [
                {'basename': 'file1'},
                {'basename': 'file2'}
            ],
            'input_number': 1,
            'input_string': 'test'
        })

        # missing files
        input_data = {
            'input_number': 1,
            'input_string': 'test'
        }
        normalized_input_data = self.data_operation._normalize_input_data(input_data)
        self.assertEqual(normalized_input_data, {
            'input_number': 1,
            'input_string': 'test'
        })
      
    def test_name(self):
        self.assertEqual(self.data_operation.name(), 'SampleDataOperation')
    
    def test_description(self):
        self.assertEqual(self.data_operation.description(), 'Testing class for DataOperation')
    
    def test_wizard_description(self):
        self.assertEqual(self.data_operation.wizard_description(), wizard_description)
    
    def test_operate(self):
        self.data_operation.operate()
        self.assertEqual(self.data_operation.get_percent_completion(), 1.0)
        self.assertEqual(self.data_operation.get_status(), 'COMPLETED')
        self.assertEqual(self.data_operation.get_output(), {'output_files': []})
    
    def test_generate_cache_key(self):
        pregenerated_cache_key = '9c8d4f7c82d357c95416c43560d0a1f66f1b2e3cb6a39c0c8a004ee162482ea7'
        self.assertEqual(self.data_operation.generate_cache_key(), pregenerated_cache_key)

    def test_set_get_output(self):
        self.data_operation.set_output({'output_files': []})
        self.assertEqual(self.data_operation.get_percent_completion(), 1.0)
        self.assertEqual(self.data_operation.get_status(), 'COMPLETED')
        self.assertEqual(self.data_operation.get_output(), {'output_files': []})

    def test_set_failed(self):
        self.data_operation.set_failed('Test message')
        self.assertEqual(self.data_operation.get_status(), 'FAILED')
        self.assertEqual(self.data_operation.get_message(), 'Test message')

class TestMedianOperation(TestCase):
    temp_median = 'datalab/datalab_session/tests/test_files/median/temp_median.fits'
    test_median = 'datalab/datalab_session/tests/test_files/median/median_1_2.fits'
    test_fits_1 = 'datalab/datalab_session/tests/test_files/median/fits_1.fits.fz'
    test_fits_2 = 'datalab/datalab_session/tests/test_files/median/fits_2.fits.fz'

    def tearDown(self):
        if os.path.exists(self.temp_median):
            os.remove(self.temp_median)
        return super().tearDown()

    @mock.patch('datalab.datalab_session.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.file_utils.get_fits')
    @mock.patch('datalab.datalab_session.data_operations.median.save_fits_and_thumbnails')
    @mock.patch('datalab.datalab_session.data_operations.median.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_fits_and_thumbnails, mock_get_fits, mock_named_tempfile):

        mock_get_fits.side_effect = [self.test_fits_1, self.test_fits_2]
        mock_named_tempfile.return_value.name = self.temp_median
        mock_create_jpgs.return_value = ('test_path', 'test_path')
        mock_save_fits_and_thumbnails.return_value = self.temp_median

        input_data = {
            'input_files': [
                {'basename': 'fits_1','source': 'local'},
                {'basename': 'fits_2','source': 'local'}
            ]
        }

        median = Median(input_data)
        median.operate()
        output = median.get_output().get('output_files')

        self.assertEqual(median.get_percent_completion(), 1.0)
        self.assertTrue(os.path.exists(output[0]))
        self.assertEqual(md5(open(self.test_median, 'rb').read()).hexdigest(), md5(open(output[0], 'rb').read()).hexdigest())

    def test_not_enough_files(self):
        input_data = {
            'input_files': [
                {'basename': 'sample_lco_fits_1'}
            ]
        }
        median = Median(input_data)
        
        with self.assertRaises(ClientAlertException):
            median.operate()

class TestRGBStackOperation(TestCase):
    temp_rgb = 'datalab/datalab_session/tests/test_files/rgb_stack/temp_rgb.fits'
    test_rgb = 'datalab/datalab_session/tests/test_files/rgb_stack/rgb_stack.fits'
    test_red = 'datalab/datalab_session/tests/test_files/rgb_stack/red.fits'
    test_green = 'datalab/datalab_session/tests/test_files/rgb_stack/green.fits'
    test_blue = 'datalab/datalab_session/tests/test_files/rgb_stack/blue.fits'

    def tearDown(self):
        if os.path.exists(self.temp_rgb):
            os.remove(self.temp_rgb)
        return super().tearDown()
    
    @mock.patch('datalab.datalab_session.data_operations.rgb_stack.save_fits_and_thumbnails')
    @mock.patch('datalab.datalab_session.data_operations.rgb_stack.create_jpgs')
    @mock.patch('datalab.datalab_session.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.rgb_stack.get_fits')
    def test_operate(self, mock_get_fits, mock_named_tempfile, mock_create_jpgs, mock_save_fits_and_thumbnails):

        mock_get_fits.side_effect = [self.test_red, self.test_green, self.test_blue]
        mock_named_tempfile.return_value.name = self.temp_rgb
        mock_create_jpgs.return_value = ('test_path', 'test_path')
        mock_save_fits_and_thumbnails.return_value = self.temp_rgb

        input_data = {
            'red_input': [{'basename': 'red_fits', 'source': 'local'}],
            'green_input': [{'basename': 'green_fits', 'source': 'local'}],
            'blue_input': [{'basename': 'blue_fits', 'source': 'local'}]
        }

        rgb = RGB_Stack(input_data)
        rgb.operate()
        output = rgb.get_output().get('output_files')

        self.assertEqual(rgb.get_percent_completion(), 1.0)
        self.assertTrue(os.path.exists(output[0]))
        self.assertEqual(md5(open(self.test_rgb, 'rb').read()).hexdigest(), md5(open(output[0], 'rb').read()).hexdigest())
