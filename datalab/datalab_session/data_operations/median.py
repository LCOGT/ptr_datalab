import logging

import numpy as np

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import crop_arrays

log = logging.getLogger()
log.setLevel(logging.INFO)


class Median(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'Median'
    
    @staticmethod
    def description():
        return """The median operation takes in 1..n input images and calculated the median value pixel-by-pixel.

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
                    'minimum': 1,
                    'maximum': 999
                }
            }
        }
    
    def operate(self):
        # Getting/Checking the Input
        input_list = self.input_data.get('input_files', [])
        if len(input_list) <= 1: raise ClientAlertException('Median needs at least 2 files')
        comment = f'Datalab Median on {", ".join([image["basename"] for image in input_list])}'
        log.info(comment)

        input_fits_list = []
        for index, input in enumerate(input_list, start=1):
            input_fits_list.append(InputDataHandler(input['basename'], input['source']))
            self.set_operation_progress(0.5 * (index / len(input_list)))

        # Creating the Median array
        cropped_data, shape = crop_arrays([image.sci_data for image in input_fits_list], flatten=True)
        median = np.median(cropped_data, axis=0, overwrite_input=True)
        median = np.reshape(median, shape)

        self.set_operation_progress(0.80)

        output = FITSOutputHandler(self.cache_key, median, self.temp, comment).create_and_save_data_products(Format.FITS)
        log.info(f'Median output: {output}')
        self.set_output(output)
