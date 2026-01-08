import logging
from typing import List

from fits_align.ident import make_transforms
from fits_align.align import affineremap
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.utils.s3_utils import save_files_to_s3
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import create_composite_jpgs, create_composite_tif, temp_file_manager


log = logging.getLogger()
log.setLevel(logging.INFO)

class Color_Image(BaseDataOperation):
    PROGRESS_STEPS = {
        'INPUT_PROCESSING': 0.4,
        'ALIGNMENT': 0.6,
        'STACKING': 0.8
    }
    
    @staticmethod
    def name():
        return 'Color Image'
    
    @staticmethod
    def description():
        return """The Color Image operation takes up to 6 images and creates a colored image by compositing them on top of each other."""

    @staticmethod
    def wizard_description():
        return {
            'name': Color_Image.name(),
            'description': Color_Image.description(),
            'category': 'image',
            'inputs': {
                'color_channels': {
                    'name': 'Color Channel',
                    'description': 'Image for a channel of the color image',
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 6,
                    'include_custom_scale': True,
                    'color_picker': True,
                },
            },
        }
    
    def _process_inputs(self, submitter, color_input_list) -> tuple[list[InputDataHandler], list[float], list[float]]:
        input_dicts: List = []
        input_handlers: List = []
        for index, input in enumerate(color_input_list, start=1):
            input_handlers.append(InputDataHandler(submitter, input['basename'], input['source']))
            self.set_operation_progress(self.PROGRESS_STEPS['INPUT_PROCESSING'] * (index / len(color_input_list)))

        # Attempt to do the image ment here
        fits_files = [handler.fits_file for handler in input_handlers]

        try:
            aligned_images = self._align_images(fits_files)
        except KeyError:
            log.info('Could not align images due to missing CAT header')
            aligned_images = fits_files

        self.set_operation_progress(self.PROGRESS_STEPS['ALIGNMENT'])

        # Now create the input_dicts needed for image composition
        for index, fits_file in enumerate(aligned_images):
            normalized_color = [color_input_list[index]['color']['r'] / 255,
                                color_input_list[index]['color']['g'] / 255,
                                color_input_list[index]['color']['b'] / 255]
            input_dicts.append({
                'fits_path': fits_file,
                'scale_algorithm': 'zscale',
                'color': normalized_color,
                'zmin': color_input_list[index]['zmin'],
                'zmax': color_input_list[index]['zmax'],
            })

        return input_dicts
    
    def _align_images(self, fits_files: list[str]) -> list[str]:
        ref_image = fits_files[0]
        images_to_align = fits_files[1:]
        identifications = make_transforms(ref_image, images_to_align)

        aligned_images = [ref_image]
        for id in identifications:
            if id.ok:
                aligned_img = affineremap(id.ukn.filepath, id.trans, outdir=self.temp)
                aligned_images.append(aligned_img)
        
        if len(aligned_images) != len(fits_files):
            log.info('could not align all images')
            return fits_files
        
        return aligned_images

    def operate(self, submitter: User):
        color_inputs = self._validate_inputs(input_key='color_channels', minimum_inputs=1)
        log.info(f"Color image operation on {', '.join([image['basename'] for image in color_inputs])}")

        input_dicts = self._process_inputs(submitter, color_inputs)

        with temp_file_manager(
            f"{self.cache_key}.tif", f"{self.cache_key}-large.jpg", f"{self.cache_key}-small.jpg",
            dir=self.temp
        ) as (tif_path, large_jpg_path, small_jpg_path):
        
            try:
                create_composite_tif(input_dicts, tif_path)
                create_composite_jpgs(input_dicts, large_jpg_path, small_jpg_path)
            except Exception as ex:
                # Catches exceptions in the fits2image methods to report back to frontend
                raise ClientAlertException(ex)

            file_paths = {
                'large_jpg_path': large_jpg_path,
                'small_jpg_path': small_jpg_path,
                'tif_path': tif_path
            }

            output = save_files_to_s3(self.cache_key, Format.IMAGE, file_paths)

        log.info(f'Color Image output: {output}')
        self.set_output(output)
        self.set_operation_progress(1.0)
        self.set_status('COMPLETED')
