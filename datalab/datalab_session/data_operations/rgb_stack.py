import logging

import numpy as np
from fits_align.ident import make_transforms
from fits_align.align import affineremap

from django.conf import settings
from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.utils.s3_utils import save_files_to_s3
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import create_jpgs, create_tif, temp_file_manager


log = logging.getLogger()
log.setLevel(logging.INFO)

class RGB_Stack(BaseDataOperation):
    REQUIRED_INPUTS = 3
    PROGRESS_STEPS = {
        'INPUT_PROCESSING': 0.4,
        'ALIGNMENT': 0.6,
        'STACKING': 0.8
    }
    
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
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 1,
                    'include_custom_scale': True,
                    'combine_custom_scale': 'rgb',
                    'filter': ['rp', 'r', 'ip', 'h-alpha']
                },
                'green_input': {
                    'name': 'Green Filter',
                    'description': 'Three images to stack their RGB values',
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 1,
                    'include_custom_scale': True,
                    'combine_custom_scale': 'rgb',
                    'filter': ['v', 'gp', 'oiii']
                },
                'blue_input': {
                    'name': 'Blue Filter',
                    'description': 'Three images to stack their RGB values',
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 1,
                    'include_custom_scale': True,
                    'combine_custom_scale': 'rgb',
                    'filter': ['b', 'sii']
                }
            },
        }
    
    def _validate_inputs(self):
        rgb_input_list = []
        for color in ['red_input', 'green_input', 'blue_input']:
            input_data = self.input_data[color][0]
            if not input_data:
                raise ClientAlertException(f'Missing {color}')
            rgb_input_list.append(input_data)

        if len(self.input_data) != self.REQUIRED_INPUTS:
            raise ClientAlertException(f'RGB stack requires exactly {self.REQUIRED_INPUTS} files')
        
        return rgb_input_list
    
    def _process_inputs(self, rgb_input_list) -> tuple[list[InputDataHandler], list[float], list[float]]:
        input_fits_list = []
        zmin_list = []
        zmax_list = []
        for index, input in enumerate(rgb_input_list, start=1):
            input_fits_list.append(InputDataHandler(input['basename'], input['source']))
            zmin_list.append(input['zmin'])
            zmax_list.append(input['zmax'])
            self.set_operation_progress(self.PROGRESS_STEPS['INPUT_PROCESSING'] * (index / len(rgb_input_list)))
        
        return input_fits_list, zmin_list, zmax_list
    
    def _align_images(self, fits_files: list[str]) -> list[str]:
        ref_image = fits_files[0]
        images_to_align = fits_files[1:]
        identifications = make_transforms(ref_image, images_to_align)

        aligned_images = [ref_image]
        for id in identifications:
            if id.ok:
                aligned_img = affineremap(id.ukn.filepath, id.trans, outdir=settings.TEMP_FITS_DIR)
                aligned_images.append(aligned_img)
        
        if len(aligned_images) != self.REQUIRED_INPUTS:
            log.info('could not align all images')
            return fits_files
        
        return aligned_images

    def operate(self):
        rgb_inputs = self._validate_inputs()
        input_handlers, zmin_list, zmax_list = self._process_inputs(rgb_inputs)
        fits_files = [handler.fits_file for handler in input_handlers]

        try:
            aligned_images = self._align_images(fits_files)
        except KeyError:
            log.info('Could not align images due to missing CAT header')
            aligned_images = fits_files

        self.set_operation_progress(self.PROGRESS_STEPS['ALIGNMENT'])

        with temp_file_manager(
            f"{self.cache_key}.tif", f"{self.cache_key}-large.jpg", f"{self.cache_key}-small.jpg",
            dir=settings.TEMP_FITS_DIR
        ) as (tif_path, large_jpg_path, small_jpg_path):
        
            try:
                create_tif(aligned_images, tif_path, color=True, zmin=zmin_list, zmax=zmax_list)
                create_jpgs(aligned_images, large_jpg_path, small_jpg_path, color=True, zmin=zmin_list, zmax=zmax_list)
            except Exception as ex:
                # Catches exceptions in the fits2image methods to report back to frontend
                raise ClientAlertException(ex)

            file_paths = {
                'large_jpg_path': large_jpg_path,
                'small_jpg_path': small_jpg_path,
                'tif_path': tif_path
            }

            output = save_files_to_s3(self.cache_key, Format.IMAGE, file_paths)

        log.info(f'RGB Stack output: {output}')
        self.set_output(output)
