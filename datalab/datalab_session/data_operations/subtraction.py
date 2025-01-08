import logging

import numpy as np

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.exceptions import ClientAlertException
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
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 999
                },
                'subtraction_file': {
                    'name': 'Subtraction File',
                    'description': 'This file will be subtracted from the input images.',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 1
                }
            },
        }

    def operate(self):

        input_files = self.input_data.get('input_files', [])
        subtraction_file_input = self.input_data.get('subtraction_file', [])

        if not subtraction_file_input: raise ClientAlertException('Missing a subtraction file')
        if len(input_files) < 1: raise ClientAlertException('Need at least one input file')

        log.info(f'Subtraction operation on {len(input_files)} files')

        subtraction_fits = InputDataHandler(subtraction_file_input[0]['basename'], subtraction_file_input[0]['source'])
        input_fits_list = []
        for index, input in enumerate(input_files, start=1):
            input_fits_list.append(InputDataHandler(input['basename'], input['source']))
            self.set_operation_progress(0.5 * (index / len(input_files)))

        outputs = []
        for index, input_image in enumerate(input_fits_list, start=1):
            # crop the input_image and subtraction_image to the same size
            input_image, subtraction_image = crop_arrays([input_image.sci_data, subtraction_fits.sci_data])

            difference_array = np.subtract(input_image, subtraction_image)

            subtraction_comment = f'Datalab Subtraction of {subtraction_file_input[0]["basename"]} subtracted from {input_files[index-1]["basename"]}'
            outputs.append(FITSOutputHandler(f'{self.cache_key}', difference_array, subtraction_comment).create_and_save_data_products(index=index))
            self.set_operation_progress(0.5 + index/len(input_fits_list) * 0.4)

        log.info(f'Subtraction output: {outputs}')
        self.set_output(outputs)
