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
    MINIMUM_NUMBER_OF_INPUTS = 2
    MAXIMUM_NUMBER_OF_INPUTS = 999
    PROGRESS_STEPS = {
        'STACKING_MIDPOINT': 0.5,
        'STACKING_PERCENTAGE_COMPLETION': 0.6,
        'STACKING_OUTPUT_PERCENTAGE_COMPLETION': 0.8,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }
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
                    'minimum': Stack.MINIMUM_NUMBER_OF_INPUTS,
                    'maximum': Stack.MAXIMUM_NUMBER_OF_INPUTS,
                }
            }
        }
        return description

    def operate(self, submitter: User):
        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS
        )
        comment= f'Datalab Stacking on {", ".join([image["basename"] for image in input_files])}'
        log.info(comment)

        input_fits_list = []
        for index, input in enumerate(input_files, start=1):
            log.info(f'this is index: {index}' f'and input: {input}')
            input_fits_list.append(InputDataHandler(submitter, input['basename'], input['source']))
            log.info(f'input fits list in normalization: {input_fits_list}')
            self.set_operation_progress(Stack.PROGRESS_STEPS['STACKING_MIDPOINT'] * (index / len(input_files)))

        cropped_data, _ = crop_arrays([image.sci_data for image in input_fits_list])
        self.set_operation_progress(Stack.PROGRESS_STEPS['STACKING_PERCENTAGE_COMPLETION'])
        stacked_sum = np.sum(cropped_data, axis=0)
        self.set_operation_progress(Stack.PROGRESS_STEPS['STACKING_OUTPUT_PERCENTAGE_COMPLETION'])
        header_template = input_fits_list[0].get_header()

        output = FITSOutputHandler(self.cache_key, stacked_sum, self.temp, comment, data_header=input_fits_list[0].sci_hdu.header.copy()).create_and_save_data_products(Format.FITS, header=header_template)

        log.info(f'Stacked output: {output}')
        self.set_output(output)
        self.set_operation_progress(Stack.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')
