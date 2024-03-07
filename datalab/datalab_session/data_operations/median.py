from datalab.datalab_session.data_operations.data_operation import BaseDataOperation


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
        num_files = len(self.input_data.get('input_files', []))

        # fetch files and store in disk memory
        for i, file in enumerate(self.input_data.get('input_files', [])):
            print(f"Processing median operation on file {file.get('basename', 'No basename found')}")
        
        # Create median fitz result file based on median of all input files

        # Loop on pixel n of each file
            # find median of pixel n
            # store median of pixel n at pixel n of new fitz

        # Generate a basename for result using a helper function

        # Store median fitz result file in S3 bitbucket
        
        # Get s3 bitbucket url
        
        # Return the output
        output = {
            'output_files': []
        }
        self.set_output(output)
