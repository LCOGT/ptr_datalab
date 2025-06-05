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

    @staticmethod
    def name():
        return 'Subtraction'

    @staticmethod
    def description():
        return """
          The Subtraction operation takes in 1..n input images and calculated the subtraction value pixel-by-pixel.
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
                    'minimum': 1,
                    'maximum': 999
                },
                'subtraction_file': {
                    'name': 'Subtraction File',
                    'description': 'This file will be subtracted from the input images.',
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 1
                }
            },
        }

    def operate(self, submitter: User):
        input_files = self.input_data.get('input_files', [])
        subtraction_file_input = self.input_data.get('subtraction_file', [])

        if not subtraction_file_input: raise ClientAlertException('Missing a subtraction file')
        if len(input_files) < 1: raise ClientAlertException('Need at least one input file')

        log.info(f'Subtraction operation on {len(input_files)} files')

        subtraction_fits = InputDataHandler(submitter, subtraction_file_input[0]['basename'], subtraction_file_input[0]['source'])
        outputs = []
        for index, input in enumerate(input_files, start=1):
            with InputDataHandler(submitter, input['basename'], input['source']) as input_image:
                self.set_operation_progress(0.9 * (index-0.5) / len(input_files))
                (input_image_data, subtraction_image), _ = crop_arrays([input_image.sci_data, subtraction_fits.sci_data])

                difference_array = np.subtract(input_image_data, subtraction_image)

                subtraction_comment = f'Datalab Subtraction of {subtraction_file_input[0]["basename"]} subtracted from {input_files[index-1]["basename"]}'
                outputs.append(FITSOutputHandler(
                    f'{self.cache_key}', difference_array, self.temp, subtraction_comment,
                    data_header=input_image.sci_hdu.header.copy()).create_and_save_data_products(Format.FITS, index=index))
                self.set_output(outputs)
                self.set_operation_progress(0.9 + index / len(input_files))

        log.info(f'Subtraction output: {outputs}')
        self.set_output(outputs)
        self.set_operation_progress(1.0)
        self.set_status('COMPLETED')
