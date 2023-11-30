from datalab.datalab_session.data_operations.data_operation import BaseDataOperation


class Median(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'Median'
    
    @staticmethod
    def description():
        return 
    '''
    The median operation takes in 1..n input images and calculated the median value pixel-by-pixel.
    The output is a median image for the n input images. This operation is commonly used for background subtraction.
    '''
    
    @staticmethod
    def wizard_description():
        return {}
    
    def operate(self, input_data):
        pass
