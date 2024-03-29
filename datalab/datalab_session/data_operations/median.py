from io import BytesIO
import logging
import os
import tempfile

import numpy as np
from astropy.io import fits
from PIL import Image

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.util import add_file_to_bucket, get_archive_from_basename, numpy_to_thumbnails

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
        cache_key = self.generate_cache_key()

        # Operation validation
        if file_count == 0:
            log.warning(f'Tried to execute median operation on {file_count} files')
            return { 'output_files': [] }
        # If cache key is already in S3 we already ran this operation on these inputs
        if False:
            # TODO: add a bucket check for cache key and then return files
            pass

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

                with fits.open(fits_url) as hdu_list:
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

            # Create thumbnails
            jpgs = numpy_to_thumbnails(median)

            # Create the Fits File
            header = fits.Header([('KEY', cache_key)])
            primary_hdu = fits.PrimaryHDU(header=header)
            image_hdu = fits.ImageHDU(median)
            hdu_list = fits.HDUList([primary_hdu, image_hdu])

            # Save Fits and Thumbnails in S3 Buckets
            add_file_to_bucket((cache_key + '/' + cache_key + '.fits'), hdu_list, 'FITS')
            add_file_to_bucket((cache_key + '/' + cache_key + '-large.jpg'), jpgs['full'], 'JPEG')
            add_file_to_bucket((cache_key + '/' + cache_key + '-small.jpg'), jpgs['thumbnail'], 'JPEG')

            # TODO: Get presigned urls for the jpgs and add to output

        # TODO: Return presigned urls as output
        output = {'output_files': []}
        self.set_percent_completion(file_count / file_count)
        self.set_output(output)
