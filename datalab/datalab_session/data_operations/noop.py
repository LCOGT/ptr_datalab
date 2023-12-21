from datalab.datalab_session.data_operations.data_operation import BaseDataOperation


class NoOperation(BaseDataOperation):
    @staticmethod
    def name():
        return 'NoOp'
    
    @staticmethod
    def description():
        return """The NoOp just returns your input images as output without doing anything!"""
    
    @staticmethod
    def wizard_description():
        return {
            'name': 'NoOp',
            'description': 'The NoOp operation returns your input images as output.\n\nIt does nothing!!!',
            'category': 'test',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': 'file',
                    'minimum': 1,
                    'maxmimum': 999
                },
                'scalar_parameter_1': {
                    'name': 'Scalar Parameter 1',
                    'description': 'This scalar parameter controls nothing',
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 25.0,
                    'default': 5.0
                },
                'string_parameter': {
                    'name': 'String Parameter',
                    'description': 'This is a string parameter',
                    'type': 'text'
                }
            }            
        }
    
    def operate(self, input_data):
        pass
