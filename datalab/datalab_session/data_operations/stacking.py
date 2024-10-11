import logging

import numpy as np

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import crop_arrays

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
        if len(input_files) <= 1: raise ClientAlertException('Stack needs at least 2 files')
        comment= f'Datalab Stacking on {", ".join([image["basename"] for image in input_files])}'
        log.info(comment)

        input_fits_list = []
        for index, input in enumerate(input_files, start=1):
            input_fits_list.append(InputDataHandler(input['basename'], input['source']))
            self.set_operation_progress(0.5 * (index / len(input_files)))

        cropped_data = crop_arrays([image.sci_data for image in input_fits_list])
        stacked_ndarray = np.stack(cropped_data, axis=2)
        self.set_operation_progress(0.6)

        # using the numpy library's sum method
        stacked_sum = np.sum(stacked_ndarray, axis=2)
        self.set_operation_progress(0.8)

        output = FITSOutputHandler(self.cache_key, stacked_sum, comment).create_and_save_data_products()

        log.info(f'Stacked output: {output}')
        self.set_output(output)
