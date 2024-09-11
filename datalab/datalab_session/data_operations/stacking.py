import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import create_fits, stack_arrays, create_jpgs
from datalab.datalab_session.s3_utils import save_fits_and_thumbnails

log = logging.getLogger()
log.setLevel(logging.INFO)


class Median(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'Stacking'
    
    @staticmethod
    def description():
        return """The stacking operation takes in 2..n input images and adds the values pixel-by-pixel.

The output is a stacked image for the n input images. This operation is commonly used for improving signal to noise."""

    @staticmethod
    def wizard_description():
        description = {
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
        return description

    def operate(self):

        input_files = self.input_data.get('input_files', [])

        if len(input_files) <= 1:
            raise ClientAlertException('Stack needs at least 2 files')

        log.info(f'Executing stacking operation on {len(input_files)} files')

        image_data_list = self.get_fits_npdata(input_files, percent=0.4, cur_percent=0.0)

        stacked_data = stack_arrays(image_data_list)

        # using the numpy library's sum method
        stacked_sum = np.sum(stacked_data, axis=2)

        fits_file = create_fits(self.cache_key, stacked_sum)

        large_jpg_path, small_jpg_path = create_jpgs(self.cache_key, fits_file)

        output_file = save_fits_and_thumbnails(self.cache_key, fits_file, large_jpg_path, small_jpg_path)

        output =  {'output_files': [output_file]}

        self.set_percent_completion(1.0)
        self.set_output(output)
        log.info(f'Stacked output: {self.get_output()}')
