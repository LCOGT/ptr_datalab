from abc import ABC, abstractmethod
from collections import namedtuple
import hashlib
import json
import os
import shutil
import logging

from django.core.cache import cache
from django.conf import settings
from datalab.datalab_session.tasks import execute_data_operation
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler

CACHE_DURATION = 60 * 60 * 24 * 30  # cache for 30 days

log = logging.getLogger()
log.setLevel(logging.INFO)

# ProgressStep will be used to tie a human readable message to the amount of time the step takes for a step
ProgressStep = namedtuple('ProgressStep', ['message', 'progress'])


class BaseDataOperation(ABC):

    def __init__(self, input_data: dict = None):
        """ The data inputs are passed in in the format described from the wizard_description """
        self.input_data = self._normalize_input_data(input_data)
        self.cache_key = self.generate_cache_key()
        self.temp = settings.TEMP_FITS_DIR # default fallback

    def _normalize_input_data(self, input_data):
        if input_data == None:
            return {}
        input_schema = self.wizard_description().get('inputs', {})
        for key, value in input_data.items():
            if input_schema.get(key, {}).get('type', '') == Format.FITS and type(value) is list:
                # If there are file type inputs with multiple files, sort them by basename since order doesn't matter
                value.sort(key=lambda x: x['basename'])
        return input_data

    @staticmethod
    @abstractmethod
    def name():
        """ A unique name for your DataOperation """

    @staticmethod
    @abstractmethod
    def description():
        """ A text description of the DataOperation, to be shown to the user """

    @staticmethod
    @abstractmethod
    def wizard_description():
        """ A json-formatted DSL describing the expected inputs for this DataOperation,
            for the frontend to create custom input widgets for it in a wizard
        """

    def _validate_file_inputs(self, input_key='input_files'):
        """ The input_key is the key in the input_files dictionary in the wizard_description that contains the list of file inputs.
            Uses the `minimum`, `maximum`, `single_filter`, and `filter_options` the input declares in the wizard_description
            to test for the input provided and ensure compliance.
        """
        input_list = self.input_data.get(input_key, [])
        input_schema = self.wizard_description().get('inputs', {}).get(input_key, {})
        input_name = input_schema.get('name', input_key)

        minimum = input_schema.get('minimum', 1)
        if not input_list or len(input_list) < minimum:
            raise ClientAlertException(f'Operation {self.name()} requires at least {minimum} {input_name} input file(s).')

        maximum = input_schema.get('maximum')
        if maximum is not None and len(input_list) > maximum:
            raise ClientAlertException(f'Operation {self.name()} accepts at most {maximum} {input_name} input file(s).')

        unique_filters = {input_file.get('filter', input_file.get('primary_optical_element', '')) or '' for input_file in input_list}

        filter_options = input_schema.get('filter_options')
        if filter_options:
            if '' in unique_filters:
                raise ClientAlertException(
                    f'Operation {self.name()} requires every {input_name} to specify the filter it was taken in, '
                    'but at least one of them does not report a filter.'
                )

            unsupported_filters = sorted(unique_filters - set(filter_options))
            if unsupported_filters:
                raise ClientAlertException(
                    f'Operation {self.name()} requires the {input_name} to be taken in one of the '
                    f'{", ".join(filter_options)} filters, but got {", ".join(unsupported_filters)}.'
                )

        if input_schema.get('single_filter'):
            if len(unique_filters) > 1:
                raise ClientAlertException(
                    f'Operation {self.name()} requires every {input_name} to be taken in the same filter, '
                    f'but the input files use {", ".join(sorted(unique_filters))}.'
                )
        return input_list

    @abstractmethod
    def operate(self, submitter):
        """ The method that performs the data operation.
            It should periodically update the percent completion during its operation.
            It should set the output and status into the cache when done.
        """
    
    def allocate_operate(self, submitter):
        """
        Wraps the operate() method, creates a unique temp directory for the operation
        """
        # Create the temp directory for the operation
        try:
            tmp_hash_path = os.path.join(self.temp, self.cache_key)
            # If tmp dir already exists, append a random hash to avoid collision
            if os.path.exists(tmp_hash_path):
                tmp_hash_path = os.path.join(tmp_hash_path, hashlib.sha256(os.urandom(8)).hexdigest())
            
            os.makedirs(tmp_hash_path)
            self.temp = tmp_hash_path
        except Exception as e:
            log.warning(f"Failed to create temp dir for operation {self.cache_key}: {e} using default {self.temp}")
        
        # Run the operation
        self.operate(submitter)

        # Clean up the temp directory
        if self.temp and os.path.exists(self.temp):
            shutil.rmtree(self.temp)

    def perform_operation(self, submitter_username):
        """ The generic method to perform the operation if its not in progress """
        status = self.get_status()
        if status == 'PENDING' or status == 'FAILED':
            self.set_status('IN_PROGRESS')
            self.set_operation_progress(0.0)
            # This asynchronous task will call the operate() method on the proper operation
            execute_data_operation.send(self.name(), self.input_data, submitter_username)

    def generate_cache_key(self) -> str:
        """ Generate a unique cache key hashed from the input_data and operation name """
        string_key = f'{self.name()}_{json.dumps(sorted(self.input_data.items()), sort_keys=True)}'
        return hashlib.sha256(string_key.encode('utf-8')).hexdigest()

    def set_status(self, status: str):
        cache.set(f'operation_{self.cache_key}_status', status, CACHE_DURATION)

    def get_status(self) -> str:
        return cache.get(f'operation_{self.cache_key}_status', 'PENDING')

    def set_message(self, message: str):
        cache.set(f'operation_{self.cache_key}_message', message, CACHE_DURATION)

    def get_message(self) -> str:
        return cache.get(f'operation_{self.cache_key}_message', '')

    def set_operation_progress(self, percent_completed: float):
        cache.set(f'operation_{self.cache_key}_progress', percent_completed, CACHE_DURATION)

    def get_operation_progress(self) -> float:
        return cache.get(f'operation_{self.cache_key}_progress', 0.0)

    def set_output(self, output, is_raw=False):
        if is_raw:
            output_data = output
        else:
            output_data = {'output_files': output if isinstance(output, list) else [output]}
        cache.set(f'operation_{self.cache_key}_output', output_data, CACHE_DURATION)

    def get_output(self) -> dict:
        return cache.get(f'operation_{self.cache_key}_output')
    
    def set_failed(self, message: str):
        self.set_status('FAILED')
        self.set_message(message)
