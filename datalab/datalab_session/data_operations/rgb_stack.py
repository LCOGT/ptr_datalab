import logging

from astropy.io import fits
import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.file_utils import create_output, get_fits, crop_arrays, create_jpgs

log = logging.getLogger()
log.setLevel(logging.INFO)


class RGB_Stack(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'RGB Stack'
    
    @staticmethod
    def description():
        return """The RGB Stack operation takes in 3 input images which have red, green, and blue filters and creates a colored image by compositing them on top of each other."""

    @staticmethod
    def wizard_description():
        return {
            'name': RGB_Stack.name(),
            'description': RGB_Stack.description(),
            'category': 'image',
            'inputs': {
                'red_input': {
                    'name': 'Red Filter',
                    'description': 'Three images to stack their RGB values',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 1,
                    'filter': ['rp', 'r']
                },
                'green_input': {
                    'name': 'Green Filter',
                    'description': 'Three images to stack their RGB values',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 1,
                    'filter': ['V', 'gp']
                },
                'blue_input': {
                    'name': 'Blue Filter',
                    'description': 'Three images to stack their RGB values',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 1,
                    'filter': ['B']
                }
            }
        }
    
    def operate(self):
        rgb_input_list = self.input_data['red_input'] + self.input_data['green_input'] + self.input_data['blue_input']

        if len(rgb_input_list) != 3:
            raise ClientAlertException('RGB stack requires exactly 3 files')
        
        log.info(f'Executing RGB Stack operation on files: {rgb_input_list}')

        fits_paths = []
        for file in rgb_input_list:
            fits_paths.append(get_fits(file.get('basename')))
            self.set_operation_progress(self.get_operation_progress() + 0.2)
        
        large_jpg_path, small_jpg_path = create_jpgs(self.cache_key, fits_paths, color=True)

        # color photos take three files, so we store it as one fits file with a 3d SCI ndarray
        arrays = [fits.open(file)['SCI'].data for file in fits_paths]
        cropped_data_list = crop_arrays(arrays)
        stacked_data = np.stack(cropped_data_list, axis=2)

        self.set_operation_progress(0.8)
        
        output =  create_output(self.cache_key, stacked_data, large_jpg=large_jpg_path, small_jpg=small_jpg_path)

        self.set_operation_progress(1.0)
        self.set_output(output)
        log.info(f'RGB Stack output: {self.get_output()}')
