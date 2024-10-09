import logging

import numpy as np

from datalab.datalab_session.data_operations.fits_file_reader import FITSFileReader
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler

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

        input_list = self.input_data.get('input_files', [])
        log.info(f'Normalization operation on {len(input_list)} file(s)')

        input_FITS_list = [FITSFileReader(input['basename'], input['source']) for input in input_list]

        output_files = []
        for index, image in enumerate(input_FITS_list, start=1):
            median = np.median(image.sci_data)
            normalized_image = image.sci_data / median

            comment = f'Datalab Normalization on file {input_list[index-1]["basename"]}'
            output = FITSOutputHandler(f'{self.cache_key}', normalized_image, comment).create_save_fits(index=index)
            output_files.append(output)
            self.set_operation_progress(0.5 + index/len(input_FITS_list) * 0.4)

        log.info(f'Normalization output: {output_files}')
        self.set_output(output_files)
