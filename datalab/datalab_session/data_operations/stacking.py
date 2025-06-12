import logging

import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import crop_arrays

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
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 999
                }
            }
        }
        return description

    def operate(self, submitter: User):
        input_files = self.input_data.get('input_files', [])
        if len(input_files) <= 1: raise ClientAlertException('Stack needs at least 2 files')
        comment= f'Datalab Stacking on {", ".join([image["basename"] for image in input_files])}'
        log.info(comment)

        input_fits_list = []
        for index, input in enumerate(input_files, start=1):
            input_fits_list.append(InputDataHandler(submitter, input['basename'], input['source']))
            self.set_operation_progress(0.5 * (index / len(input_files)))

        cropped_data, _ = crop_arrays([image.sci_data for image in input_fits_list])
        self.set_operation_progress(0.6)

        # using the numpy library's sum method
        stacked_sum = np.sum(cropped_data, axis=0)
        self.set_operation_progress(0.8)

        output = FITSOutputHandler(self.cache_key, stacked_sum, self.temp, comment).create_and_save_data_products(Format.FITS)

        log.info(f'Stacked output: {output}')
        self.set_output(output)
