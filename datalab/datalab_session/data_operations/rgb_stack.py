import logging
import tempfile

from fits2image.conversions import fits_to_jpg

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.util import add_file_to_bucket, get_fits

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

        if len(rgb_input_list) == 3:
            log.info(f'Executing RGB Stack operation on files: {rgb_input_list}')

            fits_paths = []
            for file in rgb_input_list:
                fits_paths.append(get_fits(file.get('basename')))
            
            output = self.create_jpg_output(fits_paths, percent=0.9, cur_percent=0.0, color=True)

            output =  {'output_files': output}
        else:
            output = {'output_files': []}
            raise ValueError('RGB Stack operation requires exactly 3 input files')

        self.set_percent_completion(1.0)
        self.set_output(output)
        log.info(f'RGB Stack output: {self.get_output()}')
