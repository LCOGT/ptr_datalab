import logging

import dramatiq

from datalab.datalab_session.data_operations.utils import available_operations
from datalab.datalab_session.util import get_presigned_url, key_exists

log = logging.getLogger()
log.setLevel(logging.INFO)

#TODO: Perhaps define a pipeline that can take the output of one data operation and upload to a s3 bucket, indicate success, etc...

@dramatiq.actor()
def execute_data_operation(data_operation_name: str, input_data: dict):
    operation_class = available_operations().get(data_operation_name)
    if operation_class is None:
        raise NotImplementedError("Operation not implemented!")
    else:
        operation = operation_class(input_data)
        cache_key = operation.generate_cache_key()

        # check if we've done this operation already
        if key_exists(operation.generate_cache_key()):
            log.info(f'Operation {cache_key} cached')
            
            large_jpg_url = get_presigned_url(f'{cache_key}/{cache_key}-large.jpg')
            thumbnail_jpg_url = get_presigned_url(f'{cache_key}/{cache_key}-small.jpg')

            output = {'output_files': [large_jpg_url, thumbnail_jpg_url]}

            operation.set_percent_completion(1)
            operation.set_output(output)
        else:
            operation.operate(input_data.get('input_files', []), cache_key)
