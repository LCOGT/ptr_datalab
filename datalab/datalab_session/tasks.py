import dramatiq
import logging

from datalab.datalab_session.data_operations.utils import available_operations
from requests.exceptions import RequestException

from datalab.datalab_session.exceptions import ClientAlertException

log = logging.getLogger()
log.setLevel(logging.INFO)

# Retry network connection errors 3 times, all other exceptions are not retried
def should_retry(retries_so_far, exception):
    return retries_so_far < 3 and isinstance(exception, RequestException)

@dramatiq.actor(retry_when=should_retry)
def execute_data_operation(data_operation_name: str, input_data: dict):
    operation_class = available_operations().get(data_operation_name)
    if operation_class is None:
        raise NotImplementedError("Operation not implemented!")
    else:
        try:
            operation_class(input_data).operate()
        except ClientAlertException as e:
            log.error(f"Client Error executing {data_operation_name}: {type(e).__name__}:{e}")
            operation_class(input_data).set_failed(str(e))
