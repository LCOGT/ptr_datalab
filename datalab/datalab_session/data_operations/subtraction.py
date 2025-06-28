import logging

import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import crop_arrays

log = logging.getLogger()
log.setLevel(logging.INFO)


class Subtraction(BaseDataOperation):
    MINIMUM_NUMBER_OF_INPUT_FILES = 1
    MAXIMUM_NUMBER_OF_INPUT_FILES = 999
    NUMBER_OF_SUBTRACTION_FILES = 1
    PROGRESS_STEPS = {
        'SUBTRACTION_MIDPOINT_OFFSET': 0.5,
        'SUBTRACTION_PERCENTAGE_COMPLETION': 0.8,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }
    @staticmethod
    def name():
        return 'Subtraction'

    @staticmethod
    def description():
        return """
          The Subtraction operation takes in 2..n input images and calculated the subtraction value pixel-by-pixel.
          The output is a subtraction image for the n input images. This operation is commonly used for background subtraction.
        """

    @staticmethod
    def wizard_description():
        return {
            'name': Subtraction.name(),
            'description': Subtraction.description(),
            'category': 'image',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': Format.FITS,
                    'minimum': Subtraction.MINIMUM_NUMBER_OF_INPUT_FILES,
                    'maximum': Subtraction.MAXIMUM_NUMBER_OF_INPUT_FILES,
                },
                'subtraction_file': {
                    'name': 'Subtraction File',
                    'description': 'This file will be subtracted from the input images.',
                    'type': Format.FITS,
                    'minimum': Subtraction.NUMBER_OF_SUBTRACTION_FILES,
                    'maximum': Subtraction.NUMBER_OF_SUBTRACTION_FILES,
                }
            },
        }

    def operate(self, submitter: User):
        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUT_FILES
        )

        subtraction_file_input = self._validate_inputs(
            input_key='subtraction_file',
            minimum_inputs=self.NUMBER_OF_SUBTRACTION_FILES
        )

        log.info(f'Subtraction operation on {len(input_files)} files')

        subtraction_fits = InputDataHandler(submitter, subtraction_file_input[0]['basename'], subtraction_file_input[0]['source'])
        outputs = []

        ## Processing input files
        for index, input in enumerate(input_files, start=1):
            with InputDataHandler(submitter, input['basename'], input['source']) as input_image:
                self.set_operation_progress(Subtraction.PROGRESS_STEPS['SUBTRACTION_PERCENTAGE_COMPLETION'] * (index - Subtraction.PROGRESS_STEPS['SUBTRACTION_MIDPOINT_OFFSET']) / len(input_files))
                (input_image_data, subtraction_image), _ = crop_arrays([input_image.sci_data, subtraction_fits.sci_data])
                difference_array = np.subtract(input_image_data, subtraction_image)
                subtraction_comment = f'Datalab Subtraction of {subtraction_file_input[0]["basename"]} subtracted from {input_files[index-1]["basename"]}'
                outputs.append(FITSOutputHandler(
                    f'{self.cache_key}', difference_array, self.temp, subtraction_comment,
                    data_header=input_image.sci_hdu.header.copy()).create_and_save_data_products(Format.FITS, index=index))
                self.set_output(outputs)
                self.set_operation_progress(Subtraction.PROGRESS_STEPS['SUBTRACTION_PERCENTAGE_COMPLETION'] + index / len(input_files))

        log.info(f'Subtraction output: {outputs}')
        self.set_output(outputs)
        self.set_operation_progress(Subtraction.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')
