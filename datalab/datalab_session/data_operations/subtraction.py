import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import crop_arrays, create_output

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

        if not subtraction_file_input:
            raise ClientAlertException('Missing a subtraction file')

        if len(input_files) < 1:
            raise ClientAlertException('Need at least one input file')

        log.info(f'Executing subtraction operation on {len(input_files)} files')

        input_image_data_list = self.get_fits_npdata(input_files)

        subtraction_image = self.get_fits_npdata(subtraction_file_input)[0]
        self.set_operation_progress(0.70)

        outputs = []
        for index, input_image in enumerate(input_image_data_list):
            # crop the input_image and subtraction_image to the same size
            input_image, subtraction_image = crop_arrays([input_image, subtraction_image])

            difference_array = np.subtract(input_image, subtraction_image)

            subtraction_comment = f'Product of Datalab Subtraction of {subtraction_file_input[0]["basename"]} subtracted from {input_files[index]["basename"]}'
            outputs.append(create_output(self.cache_key, difference_array, index=index, comment=subtraction_comment))
        
        self.set_operation_progress(0.90)

        self.set_output(outputs)
        log.info(f'Subtraction output: {self.get_output()}')
