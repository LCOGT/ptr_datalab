import logging
import tempfile

import numpy as np
from fits2image.conversions import fits_to_jpg

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.util import add_file_to_bucket, create_fits, stack_arrays, load_image_data_from_fits_urls

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
                    'type': 'file',
                    'minimum': 1,
                    'maxmimum': 999
                }
            }
        }
    
    def operate(self, input_files, cache_key):

        log.info(f'Executing median operation on {len(input_files)} files')

        image_data_list = load_image_data_from_fits_urls(input_files)

        self.set_percent_completion(0.4)

        stacked_data = stack_arrays(image_data_list)

        median = np.median(stacked_data, axis=2)

        hdu_list = create_fits(cache_key, median)

        # Create the output files to be stored in S3
        fits_path           = tempfile.NamedTemporaryFile(suffix=f'{cache_key}.fits').name
        large_jpg_path      = tempfile.NamedTemporaryFile(suffix=f'{cache_key}-large.jpg').name
        thumbnail_jpg_path  = tempfile.NamedTemporaryFile(suffix=f'{cache_key}-small.jpg').name
        
        hdu_list.writeto(fits_path)
        fits_to_jpg(fits_path, large_jpg_path, width=median.shape[0], height=median.shape[1])
        fits_to_jpg(fits_path, thumbnail_jpg_path)

        self.set_percent_completion(0.7)

        # Save Fits and Thumbnails in S3 Buckets
        fits_url            = add_file_to_bucket(f'{cache_key}/{cache_key}.fits', fits_path)
        large_jpg_url       = add_file_to_bucket(f'{cache_key}/{cache_key}-large.jpg', large_jpg_path)
        thumbnail_jpg_url   = add_file_to_bucket(f'{cache_key}/{cache_key}-small.jpg', thumbnail_jpg_path)

        self.set_percent_completion(0.9)

        output = {'output_files': [large_jpg_url, thumbnail_jpg_url]}

        log.info(f'Median operation output: {output}')
        self.set_percent_completion(1)
        self.set_output(output)
