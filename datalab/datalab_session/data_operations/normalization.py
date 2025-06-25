import logging

import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.exceptions import ClientAlertException

log = logging.getLogger()
log.setLevel(logging.INFO)


class Normalization(BaseDataOperation):
    MINIMUM_NUMBER_OF_INPUTS = 1
    MAXIMUM_NUMBER_OF_INPUTS = 999
    PROGRESS_MIDPOINT_OFFSET = 0.5
    PROGRESS_STEPS = {
        'INPUT_PROCESSING_PERCENTAGE_COMPLETION': 0.2,
        'NORMALIZING_PERCENTAGE_COMPLETION': 0.9,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }

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
                    'type': Format.FITS,
                    'minimum': Normalization.MINIMUM_NUMBER_OF_INPUTS,
                    'maximum': Normalization.MAXIMUM_NUMBER_OF_INPUTS,
                }
            }
        }

    def operate(self, submitter: User):
        input_list = self._validate_inputs(input_key='input_files', minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS)
        input_handlers = self._process_inputs(
            submitter,
            input_list,
            input_processing_progress=self.PROGRESS_STEPS['INPUT_PROCESSING_PERCENTAGE_COMPLETION']
        )
        log.info(f'Normalization operation on {len(input_list)} file(s)')

        output_files = []
        for index, image in enumerate(input_handlers, start=1):
            self.set_operation_progress(self.PROGRESS_STEPS['NORMALIZING_PERCENTAGE_COMPLETION'] * (index - self.PROGRESS_MIDPOINT_OFFSET) / len(input_handlers))
            median = np.median(image.sci_data)
            normalized_image = image.sci_data / median

            comment = f'Datalab Normalization on file {input_list[index-1]["basename"]}'
            output = FITSOutputHandler(
                f'{self.cache_key}',
                normalized_image,
                self.temp,
                comment,
                data_header=image.sci_hdu.header.copy()
            ).create_and_save_data_products(Format.FITS, index=index)
            output_files.append(output)
            self.set_output(output_files)
            self.set_operation_progress(self.PROGRESS_STEPS['NORMALIZING_PERCENTAGE_COMPLETION'] * index / len(input_handlers))

        log.info(f'Normalization output: {output_files}')
        self.set_output(output_files)
        self.set_operation_progress(self.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')
