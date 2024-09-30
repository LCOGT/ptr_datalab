import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import create_output, crop_arrays

log = logging.getLogger()
log.setLevel(logging.INFO)


class Stack(BaseDataOperation):
    
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
            'name': Stack.name(),
            'description': Stack.description(),
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

        image_data_list = self.get_fits_npdata(input_files)

        self.set_operation_progress(0.4)

        cropped_data = crop_arrays(image_data_list)
        stacked_data = np.stack(cropped_data, axis=2)

        self.set_operation_progress(0.6)

        # using the numpy library's sum method
        stacked_sum = np.sum(stacked_data, axis=2)
        
        self.set_operation_progress(0.8)

        output = create_output(self.cache_key, stacked_sum)

        self.set_output(output)
        log.info(f'Stacked output: {self.get_output()}')
