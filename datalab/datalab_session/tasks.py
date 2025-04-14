import dramatiq
import logging

from datalab.datalab_session.data_operations.utils import available_operations
from requests.exceptions import RequestException

from datalab.datalab_session.exceptions import ClientAlertException

log = logging.getLogger()
log.setLevel(logging.INFO)


TIME_LIMIT = 60 * 60 * 1000  # 1 hour time limit in ms


# Retry network connection errors 3 times, all other exceptions are not retried
def should_retry(retries_so_far, exception):
    return retries_so_far < 3 and isinstance(exception, RequestException)

@dramatiq.actor(retry_when=should_retry, time_limit=TIME_LIMIT)
def execute_data_operation(data_operation_name: str, input_data: dict):
    try:
        operation_class = available_operations().get(data_operation_name)
        if operation_class is None:
            raise NotImplementedError("Operation not implemented!")
        else:
            try:
                operation_class(input_data).allocate_operate()
            except ClientAlertException as error:
                log.error(f"Client Error executing {data_operation_name}: {error}")
                operation_class(input_data).set_failed(str(error))
            except Exception as error:
                log.exception(error)
                operation_class(input_data).set_failed("An unknown error ocurred, contact developers if this persists.")
    except dramatiq.middleware.TimeLimitExceeded as error:
        log.exception(error)
        available_operations().get(data_operation_name)(input_data).set_failed("The operation timed out, contact developers if this persists.")
