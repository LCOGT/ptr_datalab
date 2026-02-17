import logging

import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import crop_arrays

log = logging.getLogger()
log.setLevel(logging.INFO)


class Median(BaseDataOperation):
    MINIMUM_NUMBER_OF_INPUTS = 2
    MAXIMUM_NUMBER_OF_INPUTS = 999
    PROGRESS_STEPS = {
        'MEDIAN_MIDPOINT': 0.5,
        'MEDIAN_CALCULATION_PERCENTAGE_COMPLETION': 0.6,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }
    @staticmethod
    def name():
        return 'Median'
    
    @staticmethod
    def description():
        return """The median operation takes in 2..n input images and calculated the median value pixel-by-pixel.

The output is a median image for the n input images. This operation is commonly used for background subtraction."""

    @staticmethod
    def wizard_description():
        return {
            'name': Median.name(),
            'description': Median.description(),
            'category': 'image',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': Format.FITS,
                    'minimum': Median.MINIMUM_NUMBER_OF_INPUTS,
                    'maximum': Median.MAXIMUM_NUMBER_OF_INPUTS
                }
            }
        }
    
    def operate(self, submitter: User):
        input_list = self._validate_inputs(input_key='input_files', minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS)
        comment = f'Datalab Median on {", ".join([image["basename"] for image in input_list])}'
        log.info(comment)

        input_fits_list = []
        for index, input in enumerate(input_list, start=1):
            input_fits_list.append(InputDataHandler(submitter, input['basename'], input['source']))
            log.info(f'input fits list: {input_fits_list}')
            self.set_operation_progress(Median.PROGRESS_STEPS['MEDIAN_MIDPOINT'] * (index / len(input_list)))

        cropped_data, shape = crop_arrays([image.sci_data for image in input_fits_list], flatten=True)
        median = np.median(cropped_data, axis=0, overwrite_input=True)
        median = np.reshape(median, shape)

        self.set_operation_progress(Median.PROGRESS_STEPS['MEDIAN_CALCULATION_PERCENTAGE_COMPLETION'])
        output = FITSOutputHandler(self.cache_key, median, self.temp, comment, data_header=input_fits_list[0].sci_hdu.header.copy()).create_and_save_data_products(Format.FITS)
        log.info(f'Median output: {output}')
        self.set_output(output)
        self.set_operation_progress(Median.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')
