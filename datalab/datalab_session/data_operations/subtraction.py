import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import create_fits, create_jpgs
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
                    'maximum': 10
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
            raise ClientAlertException('subtraction file not specified')

        if len(input_files) < 1:
            raise ClientAlertException(' needs at least 1 file')

        log.info(f'Executing subtraction operation on {len(input_files)} files')

        input_image_data_list = self.get_fits_npdata(input_files)
        self.set_percent_completion(.05)

        subtraction_image = self.get_fits_npdata(subtraction_file_input)[0]
        self.set_percent_completion(.10)

        outputs = []
        for x in input_image_data_list:
            o = np.subtract(x, subtraction_image)
            fits_file = create_fits(self.cache_key, o)
            large_jpg_path, small_jpg_path = create_jpgs(self.cache_key, fits_file)
            output_file = save_fits_and_thumbnails(self.cache_key, fits_file, large_jpg_path, small_jpg_path)
            outputs.append(output_file)


        self.set_percent_completion(.90)

        output =  {'output_files': outputs}

        self.set_percent_completion(1.0)
        self.set_output(output)
        log.info(f'Subtraction output: {self.get_output()}')
