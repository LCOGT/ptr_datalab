import logging

import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.file_utils import create_fits, create_jpgs
from datalab.datalab_session.s3_utils import save_fits_and_thumbnails

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
        self.set_percent_completion(0.40)

        output_files = []
        for index, image in enumerate(image_data_list):
            median = np.median(image)
            normalized_image = image / median

            fits_file = create_fits(self.cache_key, normalized_image)
            large_jpg_path, small_jpg_path = create_jpgs(self.cache_key, fits_file)
            output_file = save_fits_and_thumbnails(self.cache_key, fits_file, large_jpg_path, small_jpg_path, index=index)
            output_files.append(output_file)

            self.set_percent_completion(self.get_percent_completion() + .40 * (index + 1) / len(input))
            
        output =  {'output_files': output_files}

        self.set_output(output)
        log.info(f'Normalization output: {self.get_output()}')
