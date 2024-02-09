from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from time import sleep
from math import ceil

class LongOperation(BaseDataOperation):
    @staticmethod
    def name():
        return 'Long'
    
    @staticmethod
    def description():
        return """The Long operation just sleeps and then returns your input images as output without doing anything"""
    
    @staticmethod
    def wizard_description():
        return {
            'name': LongOperation.name(),
            'description': LongOperation.description(),
            'category': 'test',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': 'file',
                    'minimum': 1,
                    'maxmimum': 999
                },
                'duration': {
                    'name': 'Duration',
                    'description': 'The duration of the operation',
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 99999.0,
                    'default': 60.0
                },
            }            
        }
    
    def operate(self):
        num_files = len(self.input_data.get('input_files', []))
        per_image_timeout = ceil(float(self.input_data.get('duration', 60.0)) / num_files)
        for i, file in enumerate(self.input_data.get('input_files', [])):
            print(f"Processing long operation on file {file.get('basename', 'No basename found')}")
            sleep(per_image_timeout)
            self.set_percent_completion((i+1) / num_files)
        # Done "processing" the files so set the output which sets the final status
        output = {
            'output_files': self.input_data.get('input_files', [])
        }
        self.set_output(output)
