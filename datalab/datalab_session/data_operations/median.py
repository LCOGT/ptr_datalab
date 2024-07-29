import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.util import create_fits, stack_arrays

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

        log.info(f'Executing median operation on {len(input)} files')

        if len(input) > 0:
            image_data_list = self.get_fits_npdata(input, percent=0.4, cur_percent=0.0)

            stacked_data = stack_arrays(image_data_list)

            # using the numpy library's median method
            median = np.median(stacked_data, axis=2)

            fits_file = create_fits(self.cache_key, median)

            output = self.create_jpg_output(fits_file, percent=0.6, cur_percent=0.4)

            output =  {'output_files': output}
        else:
            output = {'output_files': []}

        self.set_percent_completion(1.0)
        self.set_output(output)
        log.info(f'Median output: {self.get_output()}')
