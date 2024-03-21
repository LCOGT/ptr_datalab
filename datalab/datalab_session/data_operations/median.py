from io import BytesIO
import logging
import os
import tempfile

import numpy as np
from astropy.io import fits

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.util import store_fits_output, get_archive_from_basename

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
        file_count = len(input_files)

        if file_count == 0:
            return { 'output_files': [] }

        log.info(f'Executing median operation on {file_count} files')

        with tempfile.TemporaryDirectory() as temp_dir:
            memmap_paths = []

            for index, file_info in enumerate(input_files):
                basename = file_info.get('basename', 'No basename found')
                archive_record = get_archive_from_basename(basename)

                try:
                    fits_url = archive_record[0].get('url', 'No URL found')
                except IndexError:
                    continue

                with fits.open(fits_url, use_fsspec=True) as hdu_list:
                    data = hdu_list['SCI'].data
                    memmap_path = os.path.join(temp_dir, f'memmap_{index}.dat')
                    memmap_array = np.memmap(memmap_path, dtype=data.dtype, mode='w+', shape=data.shape)
                    memmap_array[:] = data[:]
                    memmap_paths.append(memmap_path)

                self.set_percent_completion(index / file_count)

            image_data_list = [
                np.memmap(path, dtype=np.float32, mode='r', shape=memmap_array.shape)
                for path in memmap_paths
            ]

            # Crop fits image data to be the same shape then stack
            min_shape = min(arr.shape for arr in image_data_list)
            cropped_data_list = [arr[:min_shape[0], :min_shape[1]] for arr in image_data_list]
            stacked_data = np.stack(cropped_data_list, axis=2)

            # Calculate a Median along the z axis
            median = np.median(stacked_data, axis=2)

            cache_key = self.generate_cache_key()
            header = fits.Header([('KEY', cache_key)])
            primary_hdu = fits.PrimaryHDU(header=header)
            image_hdu = fits.ImageHDU(median)
            hdu_list = fits.HDUList([primary_hdu, image_hdu])

            fits_buffer = BytesIO()
            hdu_list.writeto(fits_buffer)
            fits_buffer.seek(0)

            # Write the HDU List to the output FITS file in bitbucket
            response = store_fits_output(cache_key, fits_buffer)

        # TODO: No output yet, need to build a thumbnail service
        output = {'output_files': []}
        self.set_percent_completion(file_count / file_count)
        self.set_output(output)
