import builtins
import logging

from requests.exceptions import RequestException
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException

log=logging.getLogger()
log.setLevel(logging.INFO)

class ErrorOperation(BaseDataOperation):
  @staticmethod
  def name():
    return 'Error'
  
  @staticmethod
  def description():
    return """The Error will raise an error in the dramatiq worker!"""
  
  @staticmethod
  def wizard_description():
    return {
      'name': 'Error',
      'description': 'The Error will raise an error in the dramatiq worker!',
      'category': 'test',
      'inputs': {
        'input_files': {
            'name': 'Input Files',
            'description': 'The input files to operate on',
            'type': 'file',
            'minimum': 1,
            'maximum': 999
        },
        'Error Type': {
          'name': 'Error Type',
          'description': 'The type of error to raise, should be a python error type',
          'type': 'text',
        },
        'Error Message': {
          'name': 'Error Message',
          'description': 'The message to include with the error',
          'type': 'text',
        }
      }
    }
  
  def operate(self):
    error_type_name = self.input_data.get('Error Type')
    error_type = getattr(builtins, error_type_name, None)
    if not error_type or not issubclass(error_type, BaseException):
      raise ClientAlertException(f'Unknown Error Type: {error_type_name}')
    else:
      raise error_type(self.input_data.get('Error Message', 'No Error Message, Default Error Message!'))
