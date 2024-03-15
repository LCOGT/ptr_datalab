from io import BytesIO
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.util import store_fits_output, find_fits
import numpy as np
from astropy.io import fits
import logging
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
    
    def operate(self):
        input_files = self.input_data.get('input_files', [])
        completion_total = len(input_files)
        image_data_list = []

        log.info(f'Executing median operation on {completion_total} files')

        # fetch fits for all input data
        for index, file_info in enumerate(input_files):
            basename = file_info.get('basename', 'No basename found')
            
            fits_file = find_fits(basename)
            fits_url = fits_file[0].get('url', 'No URL found')

            with fits.open(fits_url, use_fsspec=True) as hdu_list:
                data = hdu_list['SCI'].data
                image_data_list.append(data)
                self.set_percent_completion((index) / completion_total)

        # Crop fits image data to be the same shape then stack 
        min_shape = min(arr.shape for arr in image_data_list)
        cropped_data_list = [arr[:min_shape[0], :min_shape[1]] for arr in image_data_list]
        stacked_data = np.stack(cropped_data_list, axis=2)

        # Calculate a Median along the z axis
        median = np.median(stacked_data, axis=2)

        # Create a new Fits File
        cache_key = self.generate_cache_key()
        hdr = fits.Header([('KEY', cache_key)])
        primary_hdu = fits.PrimaryHDU(header=hdr)
        image_hdu = fits.ImageHDU(median)
        hdu_list = fits.HDUList([primary_hdu, image_hdu])

        fits_buffer = BytesIO()
        hdu_list.writeto(fits_buffer)
        fits_buffer.seek(0)

        # Write the HDU List to the output FITS file in bitbucket
        response = store_fits_output(cache_key, fits_buffer)
        log.info(f'AWS response: {response}')

        # No output yet, need to build a thumbnail service
        output = {
            'output_files': []
        }
        self.set_percent_completion(completion_total / completion_total)
        self.set_output(output)
