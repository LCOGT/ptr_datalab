from unittest import mock
from hashlib import md5
import os

from django.test import TestCase

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation

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
    TEST_MEDIAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files', 'median.fits')

    def tearDown(self) -> None:
        if os.path.exists(self.TEST_MEDIAN):
            os.remove(self.TEST_MEDIAN)
        return super().tearDown()

    @mock.patch('datalab.datalab_session.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.file_utils.get_fits')
    @mock.patch('datalab.datalab_session.data_operations.median.save_fits_and_thumbnails')
    @mock.patch('datalab.datalab_session.data_operations.median.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_fits_and_thumbnails, mock_get_fits, mock_named_tempfile):

        mock_get_fits.side_effect = ['datalab/datalab_session/tests/test_files/fits_1.fits.fz', 'datalab/datalab_session/tests/test_files/fits_2.fits.fz']
        mock_named_tempfile.return_value.name = self.TEST_MEDIAN
        mock_save_fits_and_thumbnails.return_value = self.TEST_MEDIAN
        mock_create_jpgs.return_value = ('test_path', 'test_path')

        input_data = {
            'input_files': [
                {
                    'basename': 'fits_1',
                    'source': 'local'
                },
                {
                    'basename': 'fits_2',
                    'source': 'local'
                }
            ]
        }

        median = Median(input_data)
        median.operate()
        output = median.get_output().get('output_files')

        self.assertEqual(median.get_percent_completion(), 1.0)
        self.assertTrue(os.path.exists(output[0]))
        self.assertEqual(md5(open(self.TEST_MEDIAN, 'rb').read()).hexdigest(), md5(open(output[0], 'rb').read()).hexdigest())

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
    
    def testTrue(self):
        self.assertTrue(True)
