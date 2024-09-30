import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import create_fits, crop_arrays, create_jpgs
from datalab.datalab_session.s3_utils import save_fits_and_thumbnails

log = logging.getLogger()
log.setLevel(logging.INFO)


class Median(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'Median'
    
    @staticmethod
    def description():
        return """The median operation takes in 1..n input images and calculated the median value pixel-by-pixel.

The output is a median image for the n input images. This operation is commonly used for background subtraction."""

    @staticmethod
    def wizard_description():
        return {
            'name': Median.name(),
            'description': Median.description(),
            'category': 'image',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 999
                }
            }
        }
    
    def operate(self):

        input = self.input_data.get('input_files', [])

        if len(input) <= 1:
            raise ClientAlertException('Median needs at least 2 files')

        log.info(f'Executing median operation on {len(input)} files')

        image_data_list = self.get_fits_npdata(input)
        self.set_operation_progress(0.40)

        cropped_data_list = crop_arrays(image_data_list)
        stacked_data = np.stack(cropped_data_list, axis=2)

        # using the numpy library's median method
        median = np.median(stacked_data, axis=2)

        fits_file = create_fits(self.cache_key, median)

        large_jpg_path, small_jpg_path = create_jpgs(self.cache_key, fits_file)

        output_file = save_fits_and_thumbnails(self.cache_key, fits_file, large_jpg_path, small_jpg_path)

        output =  {'output_files': [output_file]}

        self.set_output(output)
        log.info(f'Median output: {self.get_output()}')
