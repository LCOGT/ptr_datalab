import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.file_utils import create_output

log = logging.getLogger()
log.setLevel(logging.INFO)


class Normalization(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'Normalization'
    
    @staticmethod
    def description():
        return """The normalize operation takes in 1..n input images and calculates each image's median value and divides every pixel by that value.

The output is a normalized image. This operation is commonly used as a precursor step for flat removal."""

    @staticmethod
    def wizard_description():
        return {
            'name': Normalization.name(),
            'description': Normalization.description(),
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
    
    def operate(self):

        input = self.input_data.get('input_files', [])

        log.info(f'Executing normalization operation on {len(input)} file(s)')

        image_data_list = self.get_fits_npdata(input)
        self.set_operation_progress(0.40)

        output_files = []
        for index, image in enumerate(image_data_list):
            median = np.median(image)
            normalized_image = image / median

            output = create_output(self.cache_key, normalized_image, index=index)
            output_files.append(output)

            self.set_operation_progress(self.get_operation_progress() + .40 * (index + 1) / len(input))

        self.set_output(output_files)
        log.info(f'Normalization output: {self.get_output()}')
