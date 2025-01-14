import logging

import numpy as np

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import crop_arrays, create_jpgs

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
                    'include_custom_scale': True,
                    'combine_custom_scale': 'rgb',
                    'filter': ['rp', 'r']
                },
                'green_input': {
                    'name': 'Green Filter',
                    'description': 'Three images to stack their RGB values',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 1,
                    'include_custom_scale': True,
                    'combine_custom_scale': 'rgb',
                    'filter': ['V', 'gp']
                },
                'blue_input': {
                    'name': 'Blue Filter',
                    'description': 'Three images to stack their RGB values',
                    'type': 'file',
                    'minimum': 1,
                    'maximum': 1,
                    'include_custom_scale': True,
                    'combine_custom_scale': 'rgb',
                    'filter': ['B']
                }
            },
        }
    
    def operate(self):
        rgb_input_list = self.input_data['red_input'] + self.input_data['green_input'] + self.input_data['blue_input']
        if len(rgb_input_list) != 3: raise ClientAlertException('RGB stack requires exactly 3 files')
        rgb_comment = f'Datalab RGB Stack on files {", ".join([image["basename"] for image in rgb_input_list])}'
        log.info(rgb_comment)

        input_fits_list = []
        zmin_list = []
        zmax_list = []
        for index, input in enumerate(rgb_input_list, start=1):
            input_fits_list.append(InputDataHandler(input['basename'], input['source']))
            zmin_list.append(input['zmin'])
            zmax_list.append(input['zmax'])
            self.set_operation_progress(0.4 * (index / len(rgb_input_list)))

        fits_file_list = [image.fits_file for image in input_fits_list]
        large_jpg_path, small_jpg_path = create_jpgs(self.cache_key, fits_file_list, color=True, zmin=zmin_list, zmax=zmax_list)
        self.set_operation_progress(0.6)

        # color photos take three files, so we store it as one fits file with a 3d SCI ndarray
        sci_data_list = [image.sci_data for image in input_fits_list]
        cropped_data_list = crop_arrays(sci_data_list)
        stacked_ndarray = np.stack(cropped_data_list, axis=2)
        self.set_operation_progress(0.8)
        
        output = FITSOutputHandler(self.cache_key, stacked_ndarray, rgb_comment).create_and_save_data_products(large_jpg_path=large_jpg_path, small_jpg_path=small_jpg_path)

        log.info(f'RGB Stack output: {output}')
        self.set_output(output)
