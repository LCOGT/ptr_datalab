import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import create_fits, create_jpgs, crop_arrays
from datalab.datalab_session.s3_utils import save_fits_and_thumbnails

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
        print(f'Input files: {input_files}')
        subtraction_file_input = self.input_data.get('subtraction_file', [])
        print(f'Subtraction file: {subtraction_file_input}')

        if not subtraction_file_input:
            raise ClientAlertException('Missing a subtraction file')

        if len(input_files) < 1:
            raise ClientAlertException('Need at least one input file')

        log.info(f'Executing subtraction operation on {len(input_files)} files')

        input_image_data_list = self.get_fits_npdata(input_files)
        self.set_percent_completion(.30)

        subtraction_image = self.get_fits_npdata(subtraction_file_input)[0]
        self.set_percent_completion(.40)

        outputs = []
        for index, input_image in enumerate(input_image_data_list):
            # crop the input_image and subtraction_image to the same size
            input_image, subtraction_image = crop_arrays([input_image, subtraction_image])

            difference_array = np.subtract(input_image, subtraction_image)

            fits_file = create_fits(self.cache_key, difference_array)
            large_jpg_path, small_jpg_path = create_jpgs(self.cache_key, fits_file)

            output_file = save_fits_and_thumbnails(self.cache_key, fits_file, large_jpg_path, small_jpg_path, index)
            outputs.append(output_file)

            self.set_percent_completion(self.get_percent_completion() + .50 * (index + 1) / len(input_files))

        output =  {'output_files': outputs}

        self.set_output(output)
        log.info(f'Subtraction output: {self.get_output()}')
