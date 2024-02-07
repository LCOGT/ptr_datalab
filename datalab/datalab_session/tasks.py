import dramatiq

from datalab.datalab_session.data_operations.data_operation import available_operations

#TODO: Perhaps define a pipeline that can take the output of one data operation and upload to a s3 bucket, indicate success, etc...

@dramatiq.actor()
def execute_data_operation(data_operation_name: str, input_data: dict, operation_id: int):
    operation_class = available_operations().get(data_operation_name)
    if operation_class is None:
        raise NotImplementedError("Operation not implemented!")
    else:
        operation_class().operate(input_data, operation_id)
