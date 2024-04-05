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
                    'maxmimum': 999
                }
            }
        }
    
    def operate(self, input_files, cache_key):

        log.info(f'Executing median operation on {len(input_files)} files')

        image_data_list = self.get_fits_npdata(input_files, percent=40.0, cur_percent=0.0)

        stacked_data = stack_arrays(image_data_list)

        # using the numpy library's median method
        median = np.median(stacked_data, axis=2)

        hdu_list = create_fits(cache_key, median)

        output = self.create_add_thumbnails_to_bucket(hdu_list, percent=60.0, cur_percent=40.0)

        output =  {'output_files': output}

        log.info(f'Median operation output: {output}')
        self.set_percent_completion(1)
        self.set_output(output)
