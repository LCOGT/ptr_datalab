from contextlib import contextmanager
from unittest import mock
import os

from astropy.io import fits
import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.rgb_stack import RGB_Stack
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.data_operations.median import Median
from datalab.datalab_session.data_operations.stacking import Stack
from datalab.datalab_session.tests.test_files.file_extended_test_case import FileExtendedTestCase
from datalab.datalab_session.utils.format import Format

wizard_description = {
            'name': 'SampleDataOperation',
            'description': 'Testing class for DataOperation',
            'category': 'test',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': Format.FITS,
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
    
    def operate(self, submitter):
        self.set_output([])


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
        self.data_operation.operate(None)
        self.assertEqual(self.data_operation.get_operation_progress(), 1.0)
        self.assertEqual(self.data_operation.get_status(), 'COMPLETED')
        self.assertEqual(self.data_operation.get_output(), {'output_files': []})
    
    def test_generate_cache_key(self):
        pregenerated_cache_key = '9c8d4f7c82d357c95416c43560d0a1f66f1b2e3cb6a39c0c8a004ee162482ea7'
        self.assertEqual(self.data_operation.generate_cache_key(), pregenerated_cache_key)

    def test_set_get_output(self):
        self.data_operation.set_output([])
        self.assertEqual(self.data_operation.get_operation_progress(), 1.0)
        self.assertEqual(self.data_operation.get_status(), 'COMPLETED')
        self.assertEqual(self.data_operation.get_output(), {'output_files': []})

    def test_set_failed(self):
        self.data_operation.set_failed('Test message')
        self.assertEqual(self.data_operation.get_status(), 'FAILED')
        self.assertEqual(self.data_operation.get_message(), 'Test message')


class TestMedianOperation(FileExtendedTestCase):
    temp_median_path = f'{test_path}temp_median.fits'
    test_median_path = f'{test_path}median/median_1_2.fits'
    test_fits_1_path = f'{test_path}fits_1.fits.fz'
    test_fits_2_path = f'{test_path}fits_2.fits.fz'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()

    @mock.patch('datalab.datalab_session.utils.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.input_data_handler.FileCache')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.FileCache', new=mock.MagicMock)
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.save_files_to_s3')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_files_to_s3, mock_file_cache, mock_named_tempfile):
        # return the test fits paths in order of the input_files instead of aws fetch
        mock_fc_instance1 = mock.MagicMock()
        mock_fc_instance1.get_fits.return_value = self.test_fits_1_path
        mock_fc_instance2 = mock.MagicMock()
        mock_fc_instance2.get_fits.return_value = self.test_fits_2_path
        mock_file_cache.side_effect = [mock_fc_instance1, mock_fc_instance2]

        # save temp output to a known path so we can test it
        mock_named_tempfile.return_value.__enter__.return_value.name = self.temp_median_path
        # avoids overwriting our output
        mock_create_jpgs.return_value.__enter__.return_value = ('test_path', 'test_path')
        # don't save to s3
        mock_save_files_to_s3.return_value = self.temp_median_path

        input_data = {
            'input_files': [
                {'basename': 'fits_1','source': 'local'},
                {'basename': 'fits_2','source': 'local'}
            ]
        }

        median = Median(input_data)
        median.operate(None)
        output = median.get_output().get('output_files')

        self.assertEqual(median.get_operation_progress(), 1.0)
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
            median.operate(None)


class TestRGBStackOperation(FileExtendedTestCase):
    temp_rgb_path = f'{test_path}temp_rgb.fits'
    test_rgb_path = f'{test_path}rgb_stack/rgb_stack.fits'
    test_red_path = f'{test_path}rgb_stack/red.fits'
    test_green_path = f'{test_path}rgb_stack/green.fits'
    test_blue_path = f'{test_path}rgb_stack/blue.fits'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()
    
    @mock.patch('datalab.datalab_session.data_operations.rgb_stack.save_files_to_s3')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.create_jpgs')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.input_data_handler.FileCache')
    def test_operate(self, mock_file_cache, mock_named_tempfile, mock_create_jpgs, mock_save_files_to_s3):
        # return the test fits paths in order of the input_files instead of aws fetch
        mock_fc_instance1 = mock.MagicMock()
        mock_fc_instance1.get_fits.return_value = self.test_red_path
        mock_fc_instance2 = mock.MagicMock()
        mock_fc_instance2.get_fits.return_value = self.test_green_path
        mock_fc_instance3 = mock.MagicMock()
        mock_fc_instance3.get_fits.return_value = self.test_blue_path
        mock_file_cache.side_effect = [mock_fc_instance1, mock_fc_instance2, mock_fc_instance3]

        # save temp output to a known path so we can test
        mock_named_tempfile.return_value.__enter__.return_value.name = self.temp_rgb_path
        # avoids overwriting our output
        mock_create_jpgs.return_value.__enter__.return_value = ('test_path', 'test_path')
        # don't save to s3
        mock_save_files_to_s3.return_value = self.temp_rgb_path

        input_data = {
            'red_input': [{'basename': 'red_fits', 'source': 'local', 'zmin': 0, 'zmax': 255}],
            'green_input': [{'basename': 'green_fits', 'source': 'local', 'zmin': 0, 'zmax': 255}],
            'blue_input': [{'basename': 'blue_fits', 'source': 'local', 'zmin': 0, 'zmax': 255}]
        }

        rgb = RGB_Stack(input_data)
        rgb.operate(None)
        output = rgb.get_output().get('output_files')

        self.assertEqual(rgb.get_operation_progress(), 1.0)
        self.assertEqual(output, [self.temp_rgb_path])


class TestStackOperation(FileExtendedTestCase):
    test_fits_1_path = f'{test_path}fits_1.fits.fz'
    test_fits_2_path = f'{test_path}fits_2.fits.fz'

    temp_stacked_path = f'{test_path}temp_stacked.fits'  
    temp_fits_1_negative_path = f'{test_path}temp_fits_1_negative.fits'
    temp_fits_2_negative_path = f'{test_path}temp_fits_2_negative.fits'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()

    @mock.patch('datalab.datalab_session.utils.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.input_data_handler.FileCache')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.FileCache', new=mock.MagicMock)
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.save_files_to_s3')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_files_to_s3, mock_file_cache, mock_named_tempfile):
        # Generate negative images
        for original, negative_path in [
            (self.test_fits_1_path, self.temp_fits_1_negative_path),
            (self.test_fits_2_path, self.temp_fits_2_negative_path)
        ]:
            hdul = fits.open(original)
            hdul['SCI'].data *= -1
            hdul.writeto(negative_path, overwrite=True)

        # Mock behavior
        mock_fc_instance1 = mock.MagicMock()
        mock_fc_instance1.get_fits.return_value = self.test_fits_1_path
        mock_fc_instance2 = mock.MagicMock()
        mock_fc_instance2.get_fits.return_value = self.test_fits_2_path
        mock_fc_instance1n = mock.MagicMock()
        mock_fc_instance1n.get_fits.return_value = self.temp_fits_1_negative_path
        mock_fc_instance2n = mock.MagicMock()
        mock_fc_instance2n.get_fits.return_value = self.temp_fits_2_negative_path
        mock_file_cache.side_effect = [mock_fc_instance1, mock_fc_instance2, mock_fc_instance1n, mock_fc_instance2n]

        mock_named_tempfile.return_value.__enter__.return_value.name = self.temp_stacked_path
        mock_create_jpgs.return_value.__enter__.return_value = ('test_path', 'test_path')
        mock_save_files_to_s3.return_value = self.temp_stacked_path

        input_data = {
            # input_data satisfies the Stack operation argument check, but the data comes from the mock_get_fits (above)
            'input_files': [
                {'basename': 'fits_1', 'source': 'local'},
                {'basename': 'fits_2', 'source': 'local'},
                {'basename': 'fits_1', 'source': 'local'},
                {'basename': 'fits_2', 'source': 'local'},
            ]
        }

        # Perform the stacking operation
        stack = Stack(input_data)
        stack.operate(None)
        output = stack.get_output().get('output_files')

        self.assertEqual(stack.get_operation_progress(), 1.0)
        self.assertEqual(self.temp_stacked_path, output[0])
        self.assertTrue(os.path.exists(self.temp_stacked_path))

        # Verify output is a blank image
        output_hdul = fits.open(self.temp_stacked_path)

        self.assertTrue(np.sum(output_hdul['SCI'].data) < 0.01)  # Changed to be close to zero because I was getting some tiny non-zero values when stacking multiple images

    def test_not_enough_files(self):
        input_data = {
            'input_files': [{'basename': 'sample_lco_fits_1'}]
        }
        median = Median(input_data)
        
        with self.assertRaises(ClientAlertException):
            median.operate(None)
