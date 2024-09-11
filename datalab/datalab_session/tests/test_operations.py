from unittest import mock
import os

import astropy.io

from astropy.io import fits

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.rgb_stack import RGB_Stack
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.data_operations.median import Median
from datalab.datalab_session.data_operations.stacking import Stack
from datalab.datalab_session.tests.test_files.file_extended_test_case import FileExtendedTestCase

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
test_path = 'datalab/datalab_session/tests/test_files/'

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


class TestDataOperation(FileExtendedTestCase):
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


class TestMedianOperation(FileExtendedTestCase):
    temp_median_path = f'{test_path}temp_median.fits'
    test_median_path = f'{test_path}median/median_1_2.fits'
    test_fits_1_path = f'{test_path}median/fits_1.fits.fz'
    test_fits_2_path = f'{test_path}median/fits_2.fits.fz'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()

    @mock.patch('datalab.datalab_session.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.file_utils.get_fits')
    @mock.patch('datalab.datalab_session.data_operations.median.save_fits_and_thumbnails')
    @mock.patch('datalab.datalab_session.data_operations.median.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_fits_and_thumbnails, mock_get_fits, mock_named_tempfile):

        # return the test fits paths in order of the input_files instead of aws fetch
        mock_get_fits.side_effect = [self.test_fits_1_path, self.test_fits_2_path]
        # save temp output to a known path so we can test it
        mock_named_tempfile.return_value.name = self.temp_median_path
        # avoids overwriting our output
        mock_create_jpgs.return_value = ('test_path', 'test_path')
        # don't save to s3
        mock_save_fits_and_thumbnails.return_value = self.temp_median_path

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
        self.assertFilesEqual(self.test_median_path, output[0])

    def test_not_enough_files(self):
        input_data = {
            'input_files': [
                {'basename': 'sample_lco_fits_1'}
            ]
        }
        median = Median(input_data)
        
        with self.assertRaises(ClientAlertException):
            median.operate()

class TestRGBStackOperation(FileExtendedTestCase):
    temp_rgb_path = f'{test_path}temp_rgb.fits'
    test_rgb_path = f'{test_path}rgb_stack/rgb_stack.fits'
    test_red_path = f'{test_path}rgb_stack/red.fits'
    test_green_path = f'{test_path}rgb_stack/green.fits'
    test_blue_path = f'{test_path}rgb_stack/blue.fits'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()
    
    @mock.patch('datalab.datalab_session.data_operations.rgb_stack.save_fits_and_thumbnails')
    @mock.patch('datalab.datalab_session.data_operations.rgb_stack.create_jpgs')
    @mock.patch('datalab.datalab_session.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.rgb_stack.get_fits')
    def test_operate(self, mock_get_fits, mock_named_tempfile, mock_create_jpgs, mock_save_fits_and_thumbnails):

        # return the test fits paths in order of the input_files instead of aws fetch
        mock_get_fits.side_effect = [self.test_red_path, self.test_green_path, self.test_blue_path]
        # save temp output to a known path so we can test
        mock_named_tempfile.return_value.name = self.temp_rgb_path
        # avoids overwriting our output
        mock_create_jpgs.return_value = ('test_path', 'test_path')
        # don't save to s3
        mock_save_fits_and_thumbnails.return_value = self.temp_rgb_path

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
        self.assertFilesEqual(self.test_rgb_path, output[0])


class TestStackOperation(FileExtendedTestCase):
    # this test should work on any fits files, so we just grab from what's there already
    test_fits_1_path = f'{test_path}median/fits_1.fits.fz'
    test_fits_2_path = f'{test_path}median/fits_2.fits.fz'

    temp_stacked_path = f'{test_path}temp_stacked.fits'  # temp output path
    temp_fits_1_negative_path = f'{test_path}temp_fits_1_negative.fits'
    temp_fits_2_negative_path = f'{test_path}temp_fits_2_negative.fits'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()

    @mock.patch('datalab.datalab_session.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.file_utils.get_fits')
    @mock.patch('datalab.datalab_session.data_operations.median.save_fits_and_thumbnails')
    @mock.patch('datalab.datalab_session.data_operations.median.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_fits_and_thumbnails, mock_get_fits, mock_named_tempfile):

        # Create a negative images using numpy
        negative_image = fits.open(self.test_fits_1_path)
        negative_image.data = np.mult(negative_image.data, -1)
        fits.writeto(self.temp_fits_1_negative_path, negative_image, overwrite=True)

        negative_image = fits.open(self.test_fits_2_path)
        negative_image.data = np.mult(negative_image.data, -1)
        fits.writeto(self.temp_fits_2_negative_path, negative_image, overwrite=True)

        # return the test fits paths in order of the input_files instead of aws fetch
        mock_get_fits.side_effect = [self.test_fits_1_path, self.test_fits_2_path,
                                     self.temp_fits_1_negative_path, self.temp_fits_2_negative_path]
        # save temp output to a known path so we can test it
        mock_named_tempfile.return_value.name = self.temp_stacked_path
        # avoids overwriting our output
        mock_create_jpgs.return_value = ('test_path', 'test_path')
        # don't save to s3
        mock_save_fits_and_thumbnails.return_value = self.temp_stacked_path

        input_data = {
            # input_data satisfies the Stack operation argument check, but the data comes from the mock_get_fits (above)
            'input_files': [
                {'basename': 'fits_1', 'source': 'local'},
                {'basename': 'fits_2', 'source': 'local'},
                {'basename': 'fits_1', 'source': 'local'},
                {'basename': 'fits_2', 'source': 'local'},
            ]
        }

        # Stack the original images and the negative images.
        # The result should be a blank image.  (x + -x = 0)
        stack = Stack(input_data)
        stack.operate()
        output = stack.get_output().get('output_files')

        # 100% completion
        self.assertEqual(stack.get_percent_completion(), 1.0)

        # test that file paths are the same
        self.assertEqual(self.temp_stacked_path, output[0])

        # output file exists
        self.assertTrue(os.path.exists(self.temp_stacked_path))

        # test that the output file (self.temp_stacked_path) is a blank image
        output_image = fits.open(self.temp_stacked_path)
        self.assertTrue(np.all(output_image.data == 0))


    def test_not_enough_files(self):
        input_data = {
            'input_files': [
                {'basename': 'sample_lco_fits_1'}
            ]
        }
        median = Median(input_data)
        
        with self.assertRaises(ClientAlertException):
            median.operate()
